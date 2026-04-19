import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Flat RaBitQ vector database following VectorChord's implementation exactly.
 *
 * Encoding (crates/rabitq/src/bit.rs :: code() + rotate.rs):
 *   1. Rotate vector via FHT + random sign flips
 *   2. Compute 4 CodeMetadata scalars: dis_u_2, factor_cnt, factor_ip, factor_err
 *   3. Pack 1-bit sign code into longs
 *
 * Query (binary::preprocess_with_distance + half_process_dot):
 *   Pass 1 – quantize query to 6-bit (simd::quantize::quantize),
 *             binarize into 6 bit-planes, accumulate via AND+popcount,
 *             compute (rough, err) dot-product estimate and prune
 *   Pass 2 – exact dot-product re-rank on survivors using normalized vectors
 *
 * Distance metric: dot product (== cosine on L2-normalized vectors).
 * Matches VectorChord's half_process_dot in bit.rs.
 */
public class BinaryVectorDB {

    static final int SEED       = 42;
    static final int QUERY_BITS = 6;    // BinaryLut BITS in VectorChord
    static final int CANDIDATES = 200;  // pass-1 survivors for exact re-rank

    private final int     D;
    private final float[] signs;
    private final VectorStore vectorStore;

    public BinaryVectorDB(int dimensions) {
        this.D           = dimensions;
        this.signs       = buildSignVector(dimensions, SEED);
        this.vectorStore = new VectorStore();
    }

    // ── Indexing ──────────────────────────────────────────────────────────────

    public boolean addVector(float[] v) { return addVector(v, null); }

    public boolean addVector(float[] v, String[] metadata) {
        try {
            BinaryVector bv = encode(v);
            if (metadata != null) bv.setMetadata(metadata);
            vectorStore.addVector(bv);
            return true;
        } catch (Exception e) {
            System.out.println(e);
            return false;
        }
    }

    /**
     * Encode a raw vector into a BinaryVector.
     * Mirrors rabitq::bit::code() + rotate_inplace() in rotate.rs.
     *
     * Input vector is L2-normalized before encoding so that dot product == cosine.
     */
    BinaryVector encode(float[] v) {
        // Normalize to unit sphere (so dot product = cosine similarity)
        float[] vn = normalizeVec(v);

        // Rotate: sign flip + FHT (rotate.rs :: rotate_inplace)
        float[] u = fhtRotate(vn);

        // Compute CodeMetadata (bit.rs :: code_metadata)
        float sumAbsX = 0, sumX2 = 0;
        int cntPos = 0, cntNeg = 0;
        for (float x : u) {
            sumAbsX += Math.abs(x);
            sumX2   += x * x;
            if (x > 0) cntPos++; else if (x < 0) cntNeg++;
        }

        float disU2   = sumX2;
        float factCnt = cntPos - cntNeg;
        float factIp  = sumX2 / sumAbsX;
        float factErr;
        {
            float disU = (float) Math.sqrt(sumX2);
            float x0   = sumAbsX / disU / (float) Math.sqrt(D);
            // bit.rs: dis_u * (1/(x0*x0) - 1).sqrt() / (n-1).sqrt()
            factErr = disU * (float) Math.sqrt(Math.max(1f / (x0 * x0) - 1f, 0f))
                      / (float) Math.sqrt(Math.max(D - 1, 1));
        }

        // Pack 1-bit sign code (bit.rs :: code_elements + binary::pack_code)
        int    numWords = (D + 63) >> 6;
        long[] code     = new long[numWords];
        for (int i = 0; i < D; i++)
            if (u[i] > 0) code[i >> 6] |= 1L << (i & 63);

        return new BinaryVector(disU2, factCnt, factIp, factErr, code);
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /**
     * Query using dot-product estimator (half_process_dot in bit.rs).
     * Input query is L2-normalized then rotated.
     *
     * Estimator (matching bit.rs :: half_process_dot):
     *   e     = k * (2*sum - qvec_sum) + b * factor_cnt
     *   rough = -e * factor_ip
     *   err   = factor_err * sqrt(dis_v_2)
     *
     * We sort by (rough + err) ascending (lower bound of -dot, i.e. worst-case score).
     * Pass 2: exact dot product re-rank.
     */
    public List<BinaryVector> query(int topK, float[] q) {
        // Normalize + rotate query (same pipeline as encode)
        float[] qn   = normalizeVec(q);
        float[] qRot = fhtRotate(qn);

        float disV2 = 0;
        for (float x : qRot) disV2 += x * x;

        // Quantize query to QUERY_BITS: simd::quantize::quantize(vector, 63.0)
        // returns (k, b, qvec_u8) where k = max((max-min)/63, 0), b = min
        float qMin = Float.MAX_VALUE, qMax = -Float.MAX_VALUE;
        for (float x : qRot) { if (x < qMin) qMin = x; if (x > qMax) qMax = x; }
        float maxVal = (1 << QUERY_BITS) - 1;  // 63
        float k = Math.max(0f, (qMax - qMin) / maxVal);
        float b = qMin;

        // qVec[i] = round((qRot[i] - b) / k)  — matches mul_add_round(lut, 1/k, -b/k)
        int[] qVec = new int[D];
        float qVecSum = 0;
        for (int i = 0; i < D; i++) {
            float val = (k == 0f) ? 0f : (qRot[i] - b) / k;
            qVec[i]  = Math.max(0, Math.min((int) maxVal, Math.round(val)));
            qVecSum += qVec[i];
        }

        // Binarize: bit-plane [bit][word]  (binary::binarize)
        int numWords = (D + 63) >> 6;
        long[][] bitPlanes = new long[QUERY_BITS][numWords];
        for (int bit = 0; bit < QUERY_BITS; bit++)
            for (int i = 0; i < D; i++)
                if (((qVec[i] >> bit) & 1) == 1)
                    bitPlanes[bit][i >> 6] |= 1L << (i & 63);

        // Pass 1: binary::accumulate + half_process_dot for every stored vector
        List<BinaryVector> vecs = vectorStore.getVectors();
        int N = vecs.size();

        float[] rough  = new float[N];
        float[] errBnd = new float[N];
        for (int i = 0; i < N; i++) {
            BinaryVector bv = vecs.get(i);

            // binary::accumulate: sum = Σ_bit popcount(code AND bitPlane[bit]) << bit
            int sum = 0;
            for (int bit = 0; bit < QUERY_BITS; bit++)
                sum += andPopcount(bv.code, bitPlanes[bit]) << bit;

            // half_process_dot (bit.rs):
            //   e     = k * (2*sum - qvec_sum) + b * factor_cnt
            //   rough = -e * factor_ip
            //   err   = factor_err * sqrt(dis_v_2)
            float e  = k * (2f * sum - qVecSum) + b * bv.factor_cnt;
            rough[i]  = -e * bv.factor_ip;
            errBnd[i] = bv.factor_err * (float) Math.sqrt(disV2);
        }

        // Sort by lower bound (rough - err) ascending = most negative dot first = best cosine first
        Integer[] idx = topKByLowerBound(rough, errBnd, Math.min(CANDIDATES, N));

        // Pass 2: exact dot product re-rank in rotated space
        // Both query and stored vectors went through the same rotation, so dot product is preserved.
        float[] exactScore = new float[idx.length];
        for (int ci = 0; ci < idx.length; ci++) {
            BinaryVector bv = vecs.get(idx[ci]);
            float sumAbsU   = bv.dis_u_2 / bv.factor_ip;
            float scale     = sumAbsU / (float) Math.sqrt(D);
            float dot = 0;
            for (int i = 0; i < D; i++) {
                float ui = ((bv.code[i >> 6] >> (i & 63)) & 1L) == 1 ? scale : -scale;
                dot += qRot[i] * ui;
            }
            exactScore[ci] = -dot; // negate: we want highest dot → lowest score
        }

        Integer[] reranked = topKAscending(exactScore, Math.min(topK, idx.length));
        List<BinaryVector> results = new ArrayList<>(reranked.length);
        for (int i : reranked) results.add(vecs.get(idx[i]));
        return results;
    }

    // ── Fast Hadamard Transform rotation ─────────────────────────────────────

    float[] fhtRotate(float[] v) {
        int p = nextPow2(D);
        float[] u = new float[p];
        for (int i = 0; i < D; i++) u[i] = v[i] * signs[i];
        fht(u);
        // Scale by 1/sqrt(p) to keep norms stable across different padded sizes
        float scale = 1f / (float) Math.sqrt(p);
        for (int i = 0; i < p; i++) u[i] *= scale;
        return Arrays.copyOf(u, D);
    }

    private static void fht(float[] u) {
        int n = u.length;
        for (int len = 1; len < n; len <<= 1)
            for (int i = 0; i < n; i += len << 1)
                for (int j = 0; j < len; j++) {
                    float a = u[i + j], b = u[i + j + len];
                    u[i + j] = a + b; u[i + j + len] = a - b;
                }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static int nextPow2(int n) {
        int p = 1; while (p < n) p <<= 1; return p;
    }

    private static int andPopcount(long[] a, long[] b) {
        int sum = 0;
        for (int i = 0; i < a.length; i++) sum += Long.bitCount(a[i] & b[i]);
        return sum;
    }

    private static Integer[] topKByLowerBound(float[] rough, float[] err, int k) {
        Integer[] idx = new Integer[rough.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(rough[a] - err[a], rough[b] - err[b]));
        return Arrays.copyOf(idx, k);
    }

    private static Integer[] topKAscending(float[] scores, int k) {
        Integer[] idx = new Integer[scores.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(scores[a], scores[b]));
        return Arrays.copyOf(idx, k);
    }

    static float[] normalizeVec(float[] v) {
        float mag = 0;
        for (float x : v) mag += x * x;
        mag = (float) Math.sqrt(mag);
        if (mag == 0f) return Arrays.copyOf(v, v.length);
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) out[i] = v[i] / mag;
        return out;
    }

    private static float[] buildSignVector(int D, int seed) {
        Random rng = new Random(seed);
        float[] s = new float[D];
        for (int i = 0; i < D; i++) s[i] = rng.nextBoolean() ? 1f : -1f;
        return s;
    }
}
