import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class VectorStore {
    private ArrayList<BinaryVector> vectorstore;

    public VectorStore(){
        this.vectorstore = new ArrayList<>();
    }

    public List<BinaryVector> getVectors(){
        return Collections.unmodifiableList(this.vectorstore);
    }

    public void addVector(BinaryVector a){
        vectorstore.add(a);
    }
}
