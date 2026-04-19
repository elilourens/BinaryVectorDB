import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * IVF (Inverted File Index) layered on top of RaBitQ binary encoding,
 * mirroring VectorChord's vchordrq approach.
 *
 * Build:
 *   1. K-means cluster the corpus into NCLUSTERS centroids (Lloyd's, on L2-normalized vecs)
 *   2. Assign each encoded BinaryVector to its nearest centroid
 *   3. Store per-cluster inverted lists
 *
 * Query:
 *   1. L2 distance from query to all centroids — O(K), cheap float32
 *   2. Select top NPROBES closest centroids
 *   3. Run RaBitQ two-pass scoring over only those clusters
 *   4. Return global top-k
 */
public class IVFBinaryVectorDB {

    static final int   QUERY_BITS = 6;
    static final int   CANDIDATES = 1000;
    static final float EPSILON    = 0.5f;

    static final int NCLUSTERS    = 256;  // IVF partitions
    static final int KMEANS_ITERS = 25;   // Lloyd's iterations
    static final int NPROBES      = 32;   // clusters to scan per query

    private final int D;

    // Set after build()
    private float[][]            centroids;              // [K][D], unit-normalized
    private List<BinaryVector>[] clusterVecs = null;    // per-cluster encoded vectors

    // Accumulated during addVector(), freed after build()
    private final List<float[]>      normalizedCorpus = new ArrayList<>();
    private final List<BinaryVector> allVectors       = new ArrayList<>();
    private final List<String[]>     allMetadata      = new ArrayList<>();

    private boolean built = false;

    public IVFBinaryVectorDB(int dimensions) {
        this.D = dimensions;
    }

    // ── Indexing ──────────────────────────────────────────────────────────────

    public boolean addVector(float[] v) { return addVector(v, null); }

    public boolean addVector(float[] v, String[] metadata) {
        if (built) throw new IllegalStateException("Cannot add vectors after build()");
        try {
            float[] vn = BinaryVectorDB.normalizeVec(v);
            normalizedCorpus.add(vn);
            allVectors.add(encodeNormalized(vn));
            allMetadata.add(metadata == null ? null : metadata.clone());
            return true;
        } catch (Exception e) {
            System.out.println(e);
            return false;
        }
    }

    /**
     * Build IVF index: K-means then assign vectors to clusters.
     * Must be called after all addVector() calls and before query().
     */
    @SuppressWarnings("unchecked")
    public void build() {
        int N = allVectors.size();
        if (N == 0) { built = true; return; }

        int K = Math.min(NCLUSTERS, N);
        float[][] corpus = normalizedCorpus.toArray(new float[0][]);

        centroids  = kMeans(corpus, K, KMEANS_ITERS);
        clusterVecs = new List[K];
        for (int c = 0; c < K; c++) clusterVecs[c] = new ArrayList<>();

        for (int i = 0; i < N; i++) {
            int c = nearestCentroid(corpus[i], centroids);
            BinaryVector bv = allVectors.get(i);
            String[] md = allMetadata.get(i);
            if (md != null) bv.setMetadata(md);
            clusterVecs[c].add(bv);
        }

        // Free corpus after clustering
        normalizedCorpus.clear();
        allVectors.clear();
        allMetadata.clear();
        built = true;
    }

    public String stats() {
        if (!built) return "not built";
        int K = centroids.length;
        int total = 0;
        for (List<BinaryVector> l : clusterVecs) total += l.size();
        return String.format("IVF(D=%d, K=%d, nprobes=%d, N=%d)", D, K, NPROBES, total);
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    public List<BinaryVector> query(int topK, float[] q) {
        if (!built) throw new IllegalStateException("Call build() before query()");

        float[] qn   = BinaryVectorDB.normalizeVec(q);
        float[] qRot = fhtRotate(qn);

        // Centroid selection: L2² distance from qn to each centroid
        int K       = centroids.length;
        int nprobes = Math.min(NPROBES, K);

        float[] centDist = new float[K];
        for (int c = 0; c < K; c++) centDist[c] = l2sq(qn, centroids[c]);

        Integer[] centIdx = new Integer[K];
        for (int i = 0; i < K; i++) centIdx[i] = i;
        Arrays.sort(centIdx, (a, b) -> Float.compare(centDist[a], centDist[b]));

        // Prepare RaBitQ query quantities (same as BinaryVectorDB.query)
        float disV2 = 0;
        for (float x : qRot) disV2 += x * x;

        float qMin = Float.MAX_VALUE, qMax = -Float.MAX_VALUE;
        for (float x : qRot) { if (x < qMin) qMin = x; if (x > qMax) qMax = x; }
        float maxVal = (1 << QUERY_BITS) - 1;
        float k = Math.max(0f, (qMax - qMin) / maxVal);
        float b = qMin;

        int[] qVec = new int[D];
        float qVecSum = 0;
        for (int i = 0; i < D; i++) {
            float val = (k == 0f) ? 0f : (qRot[i] - b) / k;
            qVec[i]   = Math.max(0, Math.min((int) maxVal, roundTiesEven(val)));
            qVecSum  += qVec[i];
        }

        int numWords = (D + 63) >> 6;
        long[][] bitPlanes = new long[QUERY_BITS][numWords];
        for (int bit = 0; bit < QUERY_BITS; bit++)
            for (int i = 0; i < D; i++)
                if (((qVec[i] >> bit) & 1) == 1)
                    bitPlanes[bit][i >> 6] |= 1L << (i & 63);

        // Score all vectors in probed clusters
        List<BinaryVector> candidates = new ArrayList<>();
        List<Float>        roughList  = new ArrayList<>();
        List<Float>        errList    = new ArrayList<>();

        for (int pi = 0; pi < nprobes; pi++) {
            for (BinaryVector bv : clusterVecs[centIdx[pi]]) {
                int sum = 0;
                for (int bit = 0; bit < QUERY_BITS; bit++)
                    sum += andPopcount(bv.code, bitPlanes[bit]) << bit;

                float e     = k * (2f * sum - qVecSum) + b * bv.factor_cnt;
                float rough = -e * bv.factor_ip;
                float err   = bv.factor_err * (float) Math.sqrt(disV2);

                candidates.add(bv);
                roughList.add(rough);
                errList.add(err);
            }
        }

        int M = candidates.size();
        if (M == 0) return new ArrayList<>();

        float[] roughArr = new float[M];
        float[] errArr   = new float[M];
        for (int i = 0; i < M; i++) { roughArr[i] = roughList.get(i); errArr[i] = errList.get(i); }

        // Pass 1: lower-bound filter
        Integer[] idx = topKByLowerBound(roughArr, errArr, Math.min(CANDIDATES, M));
        // Pass 2: re-rank by point estimate
        Integer[] reranked = topKAscending(roughArr, idx, Math.min(topK, idx.length));

        List<BinaryVector> results = new ArrayList<>(reranked.length);
        for (int i : reranked) results.add(candidates.get(i));
        return results;
    }

    // ── K-Means (Lloyd's, L2 on unit-sphere) ─────────────────────────────────

    private static float[][] kMeans(float[][] data, int K, int iters) {
        int N = data.length;
        int D = data[0].length;

        float[][] cents = new float[K][D];
        int[] perm = shuffle(N);
        for (int c = 0; c < K; c++) cents[c] = Arrays.copyOf(data[perm[c]], D);

        int[] assign = new int[N];

        for (int iter = 0; iter < iters; iter++) {
            // Assignment
            boolean changed = false;
            for (int i = 0; i < N; i++) {
                int best = nearestCentroid(data[i], cents);
                if (best != assign[i]) { assign[i] = best; changed = true; }
            }
            if (!changed && iter > 0) break;

            // Update
            float[][] sums  = new float[K][D];
            int[]     count = new int[K];
            for (int i = 0; i < N; i++) {
                int c = assign[i]; count[c]++;
                for (int d = 0; d < D; d++) sums[c][d] += data[i][d];
            }
            java.util.Random rng = new java.util.Random(iter);
            for (int c = 0; c < K; c++) {
                if (count[c] == 0) {
                    cents[c] = Arrays.copyOf(data[rng.nextInt(N)], D);
                } else {
                    float inv = 1f / count[c];
                    for (int d = 0; d < D; d++) cents[c][d] = sums[c][d] * inv;
                    // Normalize centroid to unit sphere
                    float mag = 0;
                    for (float x : cents[c]) mag += x * x;
                    mag = (float) Math.sqrt(mag);
                    if (mag > 0) for (int d = 0; d < D; d++) cents[c][d] /= mag;
                }
            }
        }
        return cents;
    }

    private static int nearestCentroid(float[] v, float[][] cents) {
        int best = 0; float bestDist = Float.MAX_VALUE;
        for (int c = 0; c < cents.length; c++) {
            float d = l2sq(v, cents[c]);
            if (d < bestDist) { bestDist = d; best = c; }
        }
        return best;
    }

    private static float l2sq(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) { float d = a[i] - b[i]; s += d * d; }
        return s;
    }

    private static int[] shuffle(int n) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) p[i] = i;
        java.util.Random rng = new java.util.Random(42);
        for (int i = n - 1; i > 0; i--) { int j = rng.nextInt(i + 1); int t = p[i]; p[i] = p[j]; p[j] = t; }
        return p;
    }

    // ── Encoding ──────────────────────────────────────────────────────────────

    private BinaryVector encodeNormalized(float[] vn) {
        float[] u = fhtRotate(vn);

        float sumAbsX = 0, sumX2 = 0;
        int cntPos = 0, cntNeg = 0;
        for (float x : u) {
            sumAbsX += Math.abs(x);
            sumX2   += x * x;
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
            factErr = disU * (float) Math.sqrt(Math.max(1f / (x0 * x0) - 1f, 0f))
                      / (float) Math.sqrt(D - 1);
        }
        int    numWords = (D + 63) >> 6;
        long[] code     = new long[numWords];
        for (int i = 0; i < D; i++)
            if ((Float.floatToRawIntBits(u[i]) & 0x80000000) == 0)
                code[i >> 6] |= 1L << (i & 63);

        return new BinaryVector(disU2, factCnt, factIp, factErr, code);
    }

    // ── Rotation (identical to BinaryVectorDB) ────────────────────────────────

    private float[] fhtRotate(float[] v) {
        int n = D, base = 31 - Integer.numberOfLeadingZeros(n), pow2 = 1 << base;
        float scale = 1f / (float) Math.sqrt(pow2);
        boolean nonPow2 = (n != pow2);
        float[] u = Arrays.copyOf(v, n);

        flip(RotationBits.BITS_0, u); fht(u, 0, pow2);
        for (int i = 0; i < pow2; i++) u[i] *= scale;
        if (nonPow2) givens(u);

        flip(RotationBits.BITS_1, u); fht(u, n - pow2, n);
        for (int i = n - pow2; i < n; i++) u[i] *= scale;
        if (nonPow2) givens(u);

        flip(RotationBits.BITS_2, u); fht(u, 0, pow2);
        for (int i = 0; i < pow2; i++) u[i] *= scale;
        if (nonPow2) givens(u);

        flip(RotationBits.BITS_3, u); fht(u, n - pow2, n);
        for (int i = n - pow2; i < n; i++) u[i] *= scale;
        if (nonPow2) givens(u);

        return u;
    }

    private static void flip(long[] bits, float[] u) {
        int n = u.length, words = n >> 6, rem = n & 63;
        for (int i = 0; i < words; i++) {
            long mask = bits[i];
            for (int j = 0; j < 64; j++)
                if (((mask >> j) & 1L) != 0) u[(i << 6) | j] = -u[(i << 6) | j];
        }
        if (rem > 0) {
            long mask = bits[words];
            for (int j = 0; j < rem; j++)
                if (((mask >> j) & 1L) != 0) u[(words << 6) | j] = -u[(words << 6) | j];
        }
    }

    private static void fht(float[] u, int from, int to) {
        int n = to - from;
        for (int len = 1; len < n; len <<= 1)
            for (int i = from; i < to; i += len << 1)
                for (int j = 0; j < len; j++) {
                    float a = u[i + j], b = u[i + j + len];
                    u[i + j] = a + b; u[i + j + len] = a - b;
                }
    }

    private static void givens(float[] u) {
        int n = u.length, m = n / 2;
        float s = 1f / (float) Math.sqrt(2.0);
        for (int i = 0; i < m; i++) {
            float a = u[i], b = u[n - m + i];
            u[i] = (a + b) * s; u[n - m + i] = (a - b) * s;
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
        return (floor & 1) == 0 ? floor : floor + 1;
    }
}
