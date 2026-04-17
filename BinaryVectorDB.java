import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class BinaryVectorDB {

    static final int RABITQ_SEED = 69;
    
    float[][] pVector;
    VectorStore vectorStore;
    public BinaryVectorDB(int dimensions) {
        this.pVector = generateOrthogonalMatrix(dimensions);
        this.vectorStore = new VectorStore();
    }

    public static void main(String[] args) {
       
       int dimensions = 0;
       if(args.length == 0){
            System.out.println("Pass Vector Dimension as arg");
       } else {
            dimensions = Integer.parseInt(args[0]);
       }
        BinaryVectorDB db = new BinaryVectorDB(dimensions);
        
    }

    public boolean addVector(float[] inputVector){
        try{
            vectorStore.addVector(RaBitQ(inputVector));
            return true;
        } catch (Exception e){
            System.out.println(e);
            return false;
        }
    }

    public List<BinaryVector> query(int top_k, float[] inputVector) {
        int D = inputVector.length;
        float sqrtD = (float) Math.sqrt(D);
        float[] rotatedQuery = rotateUnit(inputVector);

        // Binarize query for pass 1
        int numLongs = (D + 63) >> 6;
        long[] queryBits = new long[numLongs];
        for (int w = 0; w < numLongs; w++) {
            long word = 0;
            int base = w << 6;
            int lim = Math.min(64, D - base);
            for (int b = 0; b < lim; b++)
                word |= ((~Float.floatToRawIntBits(rotatedQuery[base + b]) >>> 31) & 1L) << b;
            queryBits[w] = word;
        }

        List<BinaryVector> vectorList = vectorStore.getVectors();
        int N = vectorList.size();

        // Pass 1: Hamming distance over all N vectors, pick top 100
        float[] hammingScores = new float[N];
        for (int i = 0; i < N; i++) {
            long[] bits = vectorList.get(i).getBinaryVectorDirect();
            int H = hammingDistance(bits, queryBits);
            hammingScores[i] = (float)(D - 2 * H) / D / vectorList.get(i).getInnerProduct();
        }
        Integer[] candidates = topKIndices(hammingScores, Math.min(100, N));

        // Pass 2: asymmetric re-ranking on top 100 only
        float[] refinedScores = new float[candidates.length];
        for (int ci = 0; ci < candidates.length; ci++) {
            long[] bits = vectorList.get(candidates[ci]).getBinaryVectorDirect();
            float dot = 0;
            for (int j = 0; j < D; j++)
                dot += ((bits[j >> 6] >> (j & 63)) & 1L) == 1 ? rotatedQuery[j] : -rotatedQuery[j];
            refinedScores[ci] = (dot / sqrtD) / vectorList.get(candidates[ci]).getInnerProduct();
        }
        Integer[] reranked = topKIndices(refinedScores, Math.min(top_k, candidates.length));

        List<BinaryVector> results = new ArrayList<>();
        for (int i = 0; i < reranked.length; i++)
            results.add(vectorList.get(candidates[reranked[i]]));
        return results;
    }

    private Integer[] topKIndices(float[] scores, int k) {
        Integer[] idx = new Integer[scores.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(scores[b], scores[a]));
        return Arrays.copyOf(idx, k);
    }


    public BinaryVector RaBitQ(float[] inputVector) {
        int length = inputVector.length;

        float magnitude = 0;
        for (float x : inputVector) magnitude += x * x;
        magnitude = (float) Math.sqrt(magnitude);

        float[] multiplied = rotateUnit(inputVector);  

        float innerProduct = 0;
        for (float x : multiplied) innerProduct += Math.abs(x);
        innerProduct /= (float) Math.sqrt(length);

        int numLongs = (length + 63) >> 6;
        long[] words = new long[numLongs];
        for (int w = 0; w < numLongs; w++) {
            long word = 0;
            int base = w << 6;
            int lim = Math.min(64, length - base);
            for (int b = 0; b < lim; b++)
                word |= ((~Float.floatToRawIntBits(multiplied[base + b]) >>> 31) & 1L) << b;
            words[w] = word;
        }

        return new BinaryVector(words, magnitude, innerProduct);
    }


    public int hammingDistance(long[] a, long[] b) {
        int dist = 0;
        for (int i = 0; i < a.length; i++)
            dist += Long.bitCount(a[i] ^ b[i]);
        return dist;
    }

    public float[] rotateUnit(float[] inputVector){
        int length = inputVector.length;
        float[] normalized = inputVector.clone();

        float magnitude = 0;
        for (float x : normalized) magnitude += x * x;
        magnitude = (float) Math.sqrt(magnitude);
        if (magnitude == 0) throw new IllegalArgumentException("Zero vector");
        for (int i = 0; i < length; i++) normalized[i] /= magnitude;

        return matrixMultiply(pVector, normalized);
    }

    

    public float[] matrixMultiply(float[][] p, float[] vector){
        int D = vector.length;
        float[] result = new float[D];
        for (int i = 0; i < D; i++) {
            float sum = 0;
            for (int j = 0; j < D; j++) {
                sum += p[j][i] * vector[j];
            }
            result[i] = sum;
        }
        return result;
    }

    //This is ai made I got no idea what Orthogonal even means dawg
    private float[][] generateOrthogonalMatrix(int D) {
        Random random = new Random(RABITQ_SEED);

        // Fill with random normal values
        float[][] matrix = new float[D][D];
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                matrix[i][j] = (float) random.nextGaussian();

        // Gram-Schmidt: make each row orthogonal to all previous rows, then normalise it
        float[][] Q = new float[D][D];
        for (int i = 0; i < D; i++) {
            // Start with row i from the random matrix
            float[] row = matrix[i].clone();

            // Subtract projections onto all previous Q rows
            for (int j = 0; j < i; j++) {
                float dot = 0;
                for (int k = 0; k < D; k++) dot += row[k] * Q[j][k];
                for (int k = 0; k < D; k++) row[k] -= dot * Q[j][k];
            }

            // Normalise the row to unit length
            float mag = 0;
            for (int k = 0; k < D; k++) mag += row[k] * row[k];
            mag = (float) Math.sqrt(mag);
            for (int k = 0; k < D; k++) Q[i][k] = row[k] / mag;
        }

        return Q;
    }

}
