import com.comsol.model.*;

/** Optional native COMSOL batch entry; builds the same model and solves every case. */
public class RunFinal {
  public static void main(String[] args) {
    Model m=BuildSilicaGrating.run();
    m.study("stdALL").run();
    for(String d:m.result().dataset().tags())
      System.out.println("SOLVED_DATASET "+d+" "+m.result().dataset(d).getString("solution"));
    String[] ds=m.result().dataset().tags();
    m.result().numerical("powers").set("data",ds[ds.length-1]);
    m.result().numerical("powers").set("outerinput","all");
    m.result().numerical("powers").setResult();
    double[][] values=m.result().table("tblPowers").getReal();
    System.out.println("RESULT_TABLE_ROWS="+values.length);
    if(values.length!=105) throw new IllegalStateException("Expected exactly 105 paired rows");
    for(double[] row:values) {
      if(row.length<13) throw new IllegalStateException("Missing power/identity columns");
      StringBuilder line=new StringBuilder("COMSOL_RESULT");
      for(int j=row.length-13;j<row.length;j++) line.append(',').append(Double.toString(row[j]));
      System.out.println(line.toString());
    }
    System.out.println("ALL_CONDITIONS_SOLVED=210");
  }
}
