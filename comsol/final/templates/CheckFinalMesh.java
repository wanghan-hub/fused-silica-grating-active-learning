import com.comsol.model.*;

/** Independent mesh check at both wavelength endpoints, 40 deg, all seven designs. */
public class CheckFinalMesh {
  public static void main(String[] args) {
    Model m=BuildSilicaGrating.run();
    int[] cases={3,15,18,30,33,45,48,60,63,75,78,90,93,105};
    String[] subset=new String[SweepData.LISTS.length];
    for(int j=0;j<subset.length;j++) {
      String[] all=SweepData.LISTS[j].split(" "); StringBuilder s=new StringBuilder();
      for(int c:cases) s.append(all[c-1]).append(' ');
      subset[j]=s.toString().trim();
    }
    m.study("stdALL").feature("param").set("plistarr",subset);
    double[][] baseline=null;
    double maxDiff=0, maxEnergy=0;
    for(int level=2;level<=3;level++) {
      m.param().set("mesh_scale",Integer.toString(level));
      m.study("stdALL").run();
      String[] ds=m.result().dataset().tags();
      m.result().numerical("powers").set("data",ds[ds.length-1]);
      m.result().numerical("powers").set("outerinput","all");
      m.result().numerical("powers").setResult();
      double[][] rows=m.result().table("tblPowers").getReal();
      if(rows.length!=cases.length) throw new IllegalStateException("Missing mesh-check rows");
      double[][] tail=new double[rows.length][13];
      for(int i=0;i<rows.length;i++) {
        System.arraycopy(rows[i],rows[i].length-13,tail[i],0,13);
        StringBuilder line=new StringBuilder("MESH_RESULT,"+level);
        for(double v:tail[i]) {
          if(!Double.isFinite(v)) throw new IllegalStateException("Nonfinite output");
          line.append(',').append(Double.toString(v));
        }
        System.out.println(line.toString());
        for(int j=7;j<=10;j++) if(tail[i][j]<-1e-8 || tail[i][j]>1+1e-8)
          throw new IllegalStateException("Power outside physical bounds");
        for(int j=7;j<=9;j+=2) maxEnergy=Math.max(maxEnergy,Math.abs(tail[i][j]+tail[i][j+1]-1));
        if(level==3) {
          for(int j=0;j<7;j++) if(Math.abs(tail[i][j]-baseline[i][j])>1e-8)
            throw new IllegalStateException("Mesh-check coordinate mismatch");
          for(int j=7;j<=10;j++) maxDiff=Math.max(maxDiff,Math.abs(tail[i][j]-baseline[i][j]));
        }
      }
      if(level==2) baseline=tail;
    }
    System.out.println("MESH_MAX_ABSOLUTE_DIFFERENCE="+maxDiff);
    System.out.println("MESH_MAX_ENERGY_ERROR="+maxEnergy);
    if(maxDiff>1e-5 || maxEnergy>1e-5) throw new IllegalStateException("Mesh check failed; investigate refinement before final comparison");
    System.out.println("FINAL_MESH_CHECK_PASSED=28");
  }
}
