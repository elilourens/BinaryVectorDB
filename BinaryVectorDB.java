import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
 *   Pass 2 – re-rank survivors by point estimate (rough) dropping the error bound penalty
 *
 * Distance metric: dot product (== cosine on L2-normalized vectors).
 * Matches VectorChord's half_process_dot in bit.rs.
 */
public class BinaryVectorDB {

    static final int   QUERY_BITS = 6;     // BinaryLut BITS in VectorChord
    static final int   CANDIDATES = 1000;  // pass-1 survivors for exact re-rank
    static final float EPSILON    = 0.5f;  // widens pass-1 candidate net: keep rough - EPSILON*err

    private final int     D;
    private final VectorStore vectorStore;

    public BinaryVectorDB(int dimensions) {
        this.D           = dimensions;
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
            // Match is_sign_positive / is_sign_negative: +0.0 is positive, -0.0 is negative
            if (!Float.isNaN(x)) {
                if ((Float.floatToRawIntBits(x) & 0x80000000) == 0) cntPos++;
                else cntNeg++;
            }
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
                      / (float) Math.sqrt(D - 1);
        }

        // Pack 1-bit sign code (bit.rs :: code_elements + binary::pack_code)
        // is_sign_positive: true for +0.0 and positive values (sign bit == 0)
        int    numWords = (D + 63) >> 6;
        long[] code     = new long[numWords];
        for (int i = 0; i < D; i++)
            if ((Float.floatToRawIntBits(u[i]) & 0x80000000) == 0)
                code[i >> 6] |= 1L << (i & 63);

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

        // qVec[i] = round_ties_even((qRot[i] - b) / k)  — matches mul_add_round(lut, 1/k, -b/k)
        int[] qVec = new int[D];
        float qVecSum = 0;
        for (int i = 0; i < D; i++) {
            float val = (k == 0f) ? 0f : (qRot[i] - b) / k;
            qVec[i]  = Math.max(0, Math.min((int) maxVal, roundTiesEven(val)));
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

        // Pass 1: filter by lower bound (rough - err), keep CANDIDATES
        Integer[] idx = topKByLowerBound(rough, errBnd, Math.min(CANDIDATES, N));

        // Pass 2: re-rank candidates by point estimate (rough) alone, no float32 needed
        Integer[] reranked = topKAscending(rough, idx, Math.min(topK, idx.length));
        List<BinaryVector> results = new ArrayList<>(reranked.length);
        for (int i : reranked) results.add(vecs.get(i));
        return results;
    }

    // ── Rotation (matches rotate.rs :: rotate_inplace) ───────────────────────
    //
    // 4 stages of: flip(BITS_k) -> fht(half) -> scale -> givens-walk (if non-pow2)
    // Alternates lower half (l) and upper half (r) exactly as VectorChord does.

    float[] fhtRotate(float[] v) {
        int n = D;
        int base = 31 - Integer.numberOfLeadingZeros(n); // ilog2(n)
        int pow2 = 1 << base;
        float scale = 1f / (float) Math.sqrt(pow2);
        boolean nonPow2 = (n != pow2);

        float[] u = Arrays.copyOf(v, n);

        // Stage 0: flip BITS_0, fht lower [0, pow2), scale, givens if needed
        flip(RotationBits.BITS_0, u);
        fht(u, 0, pow2);
        for (int i = 0; i < pow2; i++) u[i] *= scale;
        if (nonPow2) givens(u);

        // Stage 1: flip BITS_1, fht upper [n-pow2, n), scale, givens if needed
        flip(RotationBits.BITS_1, u);
        fht(u, n - pow2, n);
        for (int i = n - pow2; i < n; i++) u[i] *= scale;
        if (nonPow2) givens(u);

        // Stage 2: flip BITS_2, fht lower [0, pow2), scale, givens if needed
        flip(RotationBits.BITS_2, u);
        fht(u, 0, pow2);
        for (int i = 0; i < pow2; i++) u[i] *= scale;
        if (nonPow2) givens(u);

        // Stage 3: flip BITS_3, fht upper [n-pow2, n), scale, givens if needed
        flip(RotationBits.BITS_3, u);
        fht(u, n - pow2, n);
        for (int i = n - pow2; i < n; i++) u[i] *= scale;
        if (nonPow2) givens(u);

        return u;
    }

    // Flip signs using a bit-table (rotate.rs :: flip).
    // XORs the sign bit (bit 31) of each float when the corresponding table bit is 1.
    private static void flip(long[] bits, float[] u) {
        int n = u.length;
        int words = n >> 6;         // full 64-element chunks
        int rem   = n & 63;
        for (int i = 0; i < words; i++) {
            long mask = bits[i];
            for (int j = 0; j < 64; j++) {
                if (((mask >> j) & 1L) != 0)
                    u[(i << 6) | j] = -u[(i << 6) | j];
            }
        }
        if (rem > 0) {
            long mask = bits[words];
            for (int j = 0; j < rem; j++) {
                if (((mask >> j) & 1L) != 0)
                    u[(words << 6) | j] = -u[(words << 6) | j];
            }
        }
    }

    // Standard in-place FHT on u[from..to) (length must be a power of 2).
    private static void fht(float[] u, int from, int to) {
        int n = to - from;
        for (int len = 1; len < n; len <<= 1)
            for (int i = from; i < to; i += len << 1)
                for (int j = 0; j < len; j++) {
                    float a = u[i + j], b = u[i + j + len];
                    u[i + j] = a + b; u[i + j + len] = a - b;
                }
    }

    // Givens (Kaczmarz) walk: rotate.rs :: kacs_walk.
    // l = [0, m), r = [n-m, n)  where m = n/2  (skips middle element for odd n).
    private static void givens(float[] u) {
        int n = u.length;
        int m = n / 2;
        float s = 1f / (float) Math.sqrt(2.0);
        for (int i = 0; i < m; i++) {
            float a = u[i], b = u[n - m + i];
            u[i]         = (a + b) * s;
            u[n - m + i] = (a - b) * s;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static int andPopcount(long[] a, long[] b) {
        int sum = 0;
        for (int i = 0; i < a.length; i++) sum += Long.bitCount(a[i] & b[i]);
        return sum;
    }

    private static Integer[] topKByLowerBound(float[] rough, float[] err, int k) {
        Integer[] idx = new Integer[rough.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(rough[a] - EPSILON * err[a], rough[b] - EPSILON * err[b]));
        return Arrays.copyOf(idx, k);
    }

    private static Integer[] topKAscending(float[] scores, int k) {
        Integer[] idx = new Integer[scores.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(scores[a], scores[b]));
        return Arrays.copyOf(idx, k);
    }

    private static Integer[] topKAscending(float[] scores, Integer[] candidates, int k) {
        Integer[] idx = Arrays.copyOf(candidates, candidates.length);
        Arrays.sort(idx, (a, b) -> Float.compare(scores[a], scores[b]));
        return Arrays.copyOf(idx, k);
    }

    private static int roundTiesEven(float v) {
        int floor = (int) v;
        float diff = v - floor;
        if (diff < 0.5f) return floor;
        if (diff > 0.5f) return floor + 1;
        return (floor & 1) == 0 ? floor : floor + 1; // tie: round to even
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

}
