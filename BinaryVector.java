public class BinaryVector {
    private long[] binaryVector;
    private float vectorMagnitude;
    private String[] metadata;
    private float innerProduct;

    public BinaryVector(){
        
    }

    public BinaryVector(long[] binaryVector, float vectorMagnitude, float innerProduct){
        this.binaryVector = binaryVector;
        this.vectorMagnitude = vectorMagnitude;
        this.innerProduct = innerProduct;
    }

    public long[] getBinaryVector() { return binaryVector.clone(); }
    public long[] getBinaryVectorDirect() { return binaryVector; }
    public void setBinaryVector(long[] binaryVector) { this.binaryVector = binaryVector.clone(); }

    public float getVectorMagnitude() { return vectorMagnitude; }
    public void setVectorMagnitude(float vectorMagnitude) { this.vectorMagnitude = vectorMagnitude; }

    public float getInnerProduct() { return innerProduct; }
    public void setInnerProduct(float innerProduct) { this.innerProduct = innerProduct; }

    public String[] getMetadata() { return metadata == null ? null : metadata.clone(); }
    public void setMetadata(String[] metadata) { this.metadata = metadata == null ? null : metadata.clone(); }
}
