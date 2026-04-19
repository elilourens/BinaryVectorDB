public class BinaryVector {
    // Per VectorChord CodeMetadata (crates/rabitq/src/bit.rs)
    final float dis_u_2;    // ||u||^2 where u = rotated vector
    final float factor_cnt; // cnt_pos - cnt_neg
    final float factor_ip;  // dis_u_2 / sum_of_abs_x
    final float factor_err; // error bound factor

    final long[] code;      // 1-bit packed signs, length = ceil(D/64)

    private String[] metadata;

    public BinaryVector(float dis_u_2, float factor_cnt, float factor_ip, float factor_err, long[] code) {
        this.dis_u_2    = dis_u_2;
        this.factor_cnt = factor_cnt;
        this.factor_ip  = factor_ip;
        this.factor_err = factor_err;
        this.code       = code;
    }

    public String[] getMetadata() { return metadata == null ? null : metadata.clone(); }
    public void setMetadata(String[] metadata) { this.metadata = metadata == null ? null : metadata.clone(); }
}
