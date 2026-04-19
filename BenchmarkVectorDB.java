import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Benchmarks BinaryVectorDB recall and latency against exact cosine similarity search.
 */
public class BenchmarkVectorDB {

    static int NUM_VECTORS = 100_000;
    static int NUM_QUERIES = 1_000;
    static final int[] K_VALUES       = {1, 5, 10};
    static final int   MAX_K          = 10;
    static final int[] DIMS           = {384, 512, 768, 1024};
    static final int   WARMUP_QUERIES = 5;

    static final Map<Integer, String> MODEL_NAMES = new LinkedHashMap<>();
    static {
        MODEL_NAMES.put(384,  "all-MiniLM-L6-v2");
        MODEL_NAMES.put(512,  "bge-small-en-v1.5");
        MODEL_NAMES.put(768,  "all-mpnet-base-v2");
        MODEL_NAMES.put(1024, "all-roberta-large-v1");
    }

    public static void main(String[] args) {
        System.out.println("=======================================================================");
        System.out.println("       BinaryVectorDB  vs  Exact Cosine Search  --  Benchmark");
        System.out.println("=======================================================================");
        System.out.println("Mode: REAL embeddings (sentence-transformers)");
        System.out.printf("Corpus: %,d vectors  |  Queries: %d  |  k tested: %s%n%n",
                NUM_VECTORS, NUM_QUERIES, Arrays.toString(K_VALUES));

        List<String> rows = new ArrayList<>();
        for (int dim : DIMS) {
            String row = runBenchmark(dim);
            if (row != null) rows.add(row);
        }

        if (!rows.isEmpty()) {
            System.out.println("\n=======================================================================");
            System.out.println("                          Summary Table");
            System.out.println("-------+----------+-------------+-------------+-----------+-----------");
            System.out.println("  Dim  | Build ms | Exact us/q  | Binary us/q |  Speedup  | Recall@10 ");
            System.out.println("-------+----------+-------------+-------------+-----------+-----------");
            for (String row : rows) System.out.println(row);
            System.out.println("=======================================================================");
        }
    }

    static String runBenchmark(int dim) {
        System.out.printf("---  Dim = %d  --------------------------------------------------------%n", dim);

        if (!embeddingFileExists(dim, "corpus") || !embeddingFileExists(dim, "queries")) {
            System.out.printf("  Skipping: embeddings/corpus_%d.bin or embeddings/queries_%d.bin not found.%n", dim, dim);
            System.out.println("  Run generate_embeddings.py to create them.");
            System.out.println();
            return null;
        }

        System.out.printf("  Source: real embeddings (%s)%n", MODEL_NAMES.getOrDefault(dim, "?"));
        float[][] corpus  = loadEmbeddings(embeddingPath(dim, "corpus"),  NUM_VECTORS);
        float[][] queries = loadEmbeddings(embeddingPath(dim, "queries"), NUM_QUERIES);

        // ── 1a. Build flat index ──────────────────────────────────────────────
        System.out.printf("  Building flat index (%,d vectors)...", corpus.length);
        BinaryVectorDB db = new BinaryVectorDB(dim);
        long t0 = System.nanoTime();
        for (int i = 0; i < corpus.length; i++)
            db.addVector(corpus[i], new String[]{String.valueOf(i)});
        long flatBuildMs = (System.nanoTime() - t0) / 1_000_000;
        System.out.printf(" %d ms%n", flatBuildMs);

        // ── 1b. Build IVF index ───────────────────────────────────────────────
        System.out.printf("  Building IVF index (K=%d, nprobes=%d)...",
                IVFBinaryVectorDB.NCLUSTERS, IVFBinaryVectorDB.NPROBES);
        IVFBinaryVectorDB ivf = new IVFBinaryVectorDB(dim);
        long ti0 = System.nanoTime();
        for (int i = 0; i < corpus.length; i++)
            ivf.addVector(corpus[i], new String[]{String.valueOf(i)});
        ivf.build();
        long ivfBuildMs = (System.nanoTime() - ti0) / 1_000_000;
        System.out.printf(" %d ms  [%s]%n", ivfBuildMs, ivf.stats());

        // ── 2. Ground truth — exact cosine similarity ─────────────────────────
        System.out.print("  Computing ground truth (exact cosine search)...");
        float[][] normCorpus = normalizeAll(corpus);
        int[][]   groundTruth = new int[queries.length][MAX_K];

        for (int qi = 0; qi < WARMUP_QUERIES; qi++)
            exactTopK(normCorpus, normalizeVec(queries[qi]), MAX_K);

        long t1 = System.nanoTime();
        for (int qi = 0; qi < queries.length; qi++)
            groundTruth[qi] = exactTopK(normCorpus, normalizeVec(queries[qi]), MAX_K);
        long exactNs = System.nanoTime() - t1;
        System.out.printf(" %.1f ms%n", exactNs / 1e6);

        // ── 3a. Flat binary DB queries ────────────────────────────────────────
        System.out.print("  Running flat binary queries...");
        int[][] flatResults = new int[queries.length][MAX_K];
        for (int qi = 0; qi < WARMUP_QUERIES; qi++) db.query(MAX_K, queries[qi]);
        long t2 = System.nanoTime();
        for (int qi = 0; qi < queries.length; qi++) {
            List<BinaryVector> results = db.query(MAX_K, queries[qi]);
            for (int k = 0; k < results.size(); k++)
                flatResults[qi][k] = Integer.parseInt(results.get(k).getMetadata()[0]);
        }
        long flatNs = System.nanoTime() - t2;
        System.out.printf(" %.1f ms%n", flatNs / 1e6);

        // ── 3b. IVF binary DB queries ─────────────────────────────────────────
        System.out.print("  Running IVF binary queries...");
        int[][] ivfResults = new int[queries.length][MAX_K];
        for (int qi = 0; qi < WARMUP_QUERIES; qi++) ivf.query(MAX_K, queries[qi]);
        long ti2 = System.nanoTime();
        for (int qi = 0; qi < queries.length; qi++) {
            List<BinaryVector> results = ivf.query(MAX_K, queries[qi]);
            for (int k = 0; k < results.size(); k++)
                ivfResults[qi][k] = Integer.parseInt(results.get(k).getMetadata()[0]);
        }
        long ivfNs = System.nanoTime() - ti2;
        System.out.printf(" %.1f ms%n", ivfNs / 1e6);

        // ── 4. Recall & stats ─────────────────────────────────────────────────
        double[] flatRecalls = new double[K_VALUES.length];
        double[] ivfRecalls  = new double[K_VALUES.length];
        for (int ki = 0; ki < K_VALUES.length; ki++) {
            flatRecalls[ki] = computeRecall(groundTruth, flatResults, K_VALUES[ki]);
            ivfRecalls[ki]  = computeRecall(groundTruth, ivfResults,  K_VALUES[ki]);
        }

        double exactUs = exactNs / 1e3 / queries.length;
        double flatUs  = flatNs  / 1e3 / queries.length;
        double ivfUs   = ivfNs   / 1e3 / queries.length;

        System.out.printf("  Flat build: %d ms  |  IVF build: %d ms%n", flatBuildMs, ivfBuildMs);
        System.out.printf("  Exact avg:  %.1f us/q%n", exactUs);
        System.out.printf("  Flat  avg:  %.1f us/q  (%.2fx speedup)%n", flatUs,  (double)exactNs/flatNs);
        System.out.printf("  IVF   avg:  %.1f us/q  (%.2fx speedup)%n", ivfUs,   (double)exactNs/ivfNs);
        for (int ki = 0; ki < K_VALUES.length; ki++)
            System.out.printf("  Recall@%-2d   flat=%.4f  ivf=%.4f%n",
                    K_VALUES[ki], flatRecalls[ki], ivfRecalls[ki]);
        System.out.println();

        return String.format("| %5d | %8d | %11.1f | %9.1f | %9.1f | flat=%.4f ivf=%.4f |",
                dim, flatBuildMs, exactUs, flatUs, ivfUs,
                flatRecalls[K_VALUES.length - 1], ivfRecalls[K_VALUES.length - 1]);
    }

    // ── Embedding file I/O ────────────────────────────────────────────────────

    static String embeddingPath(int dim, String split) {
        return "embeddings" + File.separator + split + "_" + dim + ".bin";
    }

    static boolean embeddingFileExists(int dim, String split) {
        return new File(embeddingPath(dim, split)).exists();
    }

    static float[][] loadEmbeddings(String path, int maxVectors) {
        try (FileChannel ch = FileChannel.open(Paths.get(path), StandardOpenOption.READ)) {
            ByteBuffer header = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            ch.read(header);
            header.flip();
            int fileN = header.getInt();
            int fileD = header.getInt();
            int n = Math.min(fileN, maxVectors);

            ByteBuffer data = ByteBuffer.allocate(n * fileD * 4).order(ByteOrder.LITTLE_ENDIAN);
            ch.read(data);
            data.flip();

            float[][] vecs = new float[n][fileD];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < fileD; j++)
                    vecs[i][j] = data.getFloat();

            System.out.printf("  Loaded %d x %d from %s%n", n, fileD, path);
            return vecs;
        } catch (IOException e) {
            throw new RuntimeException("Failed to load " + path + ": " + e.getMessage(), e);
        }
    }

    // ── Vector utilities ──────────────────────────────────────────────────────

    static float[][] normalizeAll(float[][] vecs) {
        float[][] out = new float[vecs.length][];
        for (int i = 0; i < vecs.length; i++) out[i] = normalizeVec(vecs[i]);
        return out;
    }

    static float[] normalizeVec(float[] v) {
        float mag = 0;
        for (float x : v) mag += x * x;
        mag = (float) Math.sqrt(mag);
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) out[i] = v[i] / mag;
        return out;
    }

    static int[] exactTopK(float[][] corpus, float[] query, int k) {
        int n   = corpus.length;
        int dim = query.length;
        float[] scores = new float[n];
        for (int i = 0; i < n; i++) {
            float dot = 0;
            for (int j = 0; j < dim; j++) dot += corpus[i][j] * query[j];
            scores[i] = dot;
        }
        int[]  topK     = new int[k];
        float  minScore = Float.NEGATIVE_INFINITY;
        int    minPos   = 0;

        for (int i = 0; i < n; i++) {
            if (i < k) {
                topK[i] = i;
                if (scores[i] < minScore || i == 0) { minScore = scores[i]; minPos = i; }
                if (i == k - 1) {
                    minScore = Float.MAX_VALUE;
                    for (int p = 0; p < k; p++)
                        if (scores[topK[p]] < minScore) { minScore = scores[topK[p]]; minPos = p; }
                }
            } else if (scores[i] > minScore) {
                topK[minPos] = i;
                minScore = Float.MAX_VALUE;
                for (int p = 0; p < k; p++)
                    if (scores[topK[p]] < minScore) { minScore = scores[topK[p]]; minPos = p; }
            }
        }
        return topK;
    }

    static double computeRecall(int[][] groundTruth, int[][] results, int k) {
        int total = 0, hit = 0;
        for (int qi = 0; qi < groundTruth.length; qi++) {
            Set<Integer> gtSet = new HashSet<>(k * 2);
            for (int i = 0; i < k; i++) gtSet.add(groundTruth[qi][i]);
            for (int i = 0; i < k; i++) if (gtSet.contains(results[qi][i])) hit++;
            total += k;
        }
        return (double) hit / total;
    }
}
