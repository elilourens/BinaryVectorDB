public class TestRotation {
    public static void main(String[] args) {
        BinaryVectorDB db = new BinaryVectorDB(3);
        float[] v = {2.0f, 3.0f, 4.0f};
        float[] r = db.fhtRotate(v);
        System.out.printf("Got:      [%.7f, %.7f, %.7f]%n", r[0], r[1], r[2]);
        System.out.println("Expected: [3.9819170, 1.8043789, 3.1446066]");
    }
}
