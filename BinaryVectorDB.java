import java.util.ArrayList;
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

    public List<BinaryVector> query(int top_k, float[] inputVector){
        BinaryVector queryVector = RaBitQ(inputVector);
        List<BinaryVector> vectorList = vectorStore.getVectors();

        List<int[]> distances = new ArrayList<>();
        for(int i = 0; i < vectorList.size(); i++){
            int distance = hammingDistance(vectorList.get(i).getBinaryVector(), queryVector.getBinaryVector());
            distances.add(new int[]{distance, i});
        }

        distances.sort((a, b) -> Integer.compare(a[0], b[0]));

        List<BinaryVector> results = new ArrayList<>();
        for(int i = 0; i < Math.min(top_k, distances.size()); i++){
            results.add(vectorList.get(distances.get(i)[1]));
        }
        return results;
    }

    public BinaryVector RaBitQ(float[] inputVector){
        float[] normalized = inputVector.clone();

        int length = inputVector.length;

        float magnitude = 0;
        for(int i = 0; i < length; i++) magnitude += normalized[i] * normalized[i];
        magnitude = (float) Math.sqrt(magnitude);
        if (magnitude == 0) throw new IllegalArgumentException("Zero vector");
        for(int i = 0; i < length; i++) normalized[i] /= magnitude;

        float[] multiplied = matrixMultiply(pVector, normalized);

        int numLongs = (length + 63) >> 6;
        long[] words = new long[numLongs];
        for (int w = 0; w < numLongs; w++) {
            long word = 0;
            int base = w << 6;
            int lim = Math.min(64, length - base);
            for (int b = 0; b < lim; b++) {
                word |= ((~Float.floatToRawIntBits(multiplied[base + b]) >>> 31) & 1L) << b;
            }
            words[w] = word;
        }

        return new BinaryVector(words, magnitude);
    }

    public int hammingDistance(long[] a, long[] b) {
        int dist = 0;
        for (int i = 0; i < a.length; i++)
            dist += Long.bitCount(a[i] ^ b[i]);
        return dist;
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
