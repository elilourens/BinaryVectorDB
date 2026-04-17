public class BinaryVector {
    private long[] binaryVector;
    private float vectorMagnitude;
    private String[] metadata;

    public BinaryVector(){
        
    }

    public BinaryVector(long[] binaryVector, float vectorMagnitude){
        this.binaryVector = binaryVector;
        this.vectorMagnitude = vectorMagnitude;
    }

    public long[] getBinaryVector() { return binaryVector.clone(); }
    public void setBinaryVector(long[] binaryVector) { this.binaryVector = binaryVector.clone(); }

    public float getVectorMagnitude() { return vectorMagnitude; }
    public void setVectorMagnitude(float vectorMagnitude) { this.vectorMagnitude = vectorMagnitude; }

    public String[] getMetadata() { return metadata == null ? null : metadata.clone(); }
    public void setMetadata(String[] metadata) { this.metadata = metadata == null ? null : metadata.clone(); }
}
