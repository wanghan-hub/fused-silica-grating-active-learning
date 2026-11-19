import com.comsol.model.*;
import com.comsol.model.util.*;

/** COMSOL 6.1 native FEM model. No RCWA data is read or used. */
public class BuildSilicaGrating {
  // These paired sweep lists are generated from the preserved request CSV.
  static final String[] NAMES = {"case_id", "period_nm", "height_nm", "top_duty", "bottom_duty", "wavelength_nm", "angle_deg"};
  static final String[] LISTS = SweepData.LISTS;
  public static void main(String[] args) { run(); }
  public static Model run() {
    Model m = ModelUtil.create("Model");
    m.label("Silica trapezoidal grating - initial independent FEM validation");
    m.comments("COMSOL 6.1. Air incidence, semi-infinite silica represented by a homogeneous substrate buffer and outgoing periodic port. Exact continuous trapezoid, lossless Sellmeier dispersion. TE and TM have separate uncoupled physics in one study with 45 paired conditions. All R and T are independently evaluated from port solutions. No RCWA labels or power renormalization.");
    m.param().set("case_id", "1");
    m.param().set("period_nm", "180");
    m.param().set("height_nm", "0");
    m.param().set("top_duty", "0.25");
    m.param().set("bottom_duty", "0.75");
    m.param().set("wavelength_nm", "450");
    m.param().set("angle_deg", "0");
    m.param().set("mesh_scale", "2", "Verified finer mesh; increase to 3 for independent refinement");
    m.param().set("buffer_nm", "1000", "Homogeneous air and silica distances from structure to ports");
    m.param().set("P", "period_nm*1[nm]");
    m.param().set("H", "height_nm*1[nm]");
    m.param().set("Hg", "if(height_nm>0,height_nm,20)*1[nm]", "At h=0 this partition is AIR, not a silica ridge");
    m.param().set("B", "buffer_nm*1[nm]");
    m.param().set("lam", "wavelength_nm*1[nm]");
    m.param().set("alpha", "angle_deg*pi/180");
    m.param().set("lu", "lam/1[um]");
    m.param().set("nSi", "sqrt(1+0.6961663*lu^2/(lu^2-0.0684043^2)+0.4079426*lu^2/(lu^2-0.1162414^2)+0.8974794*lu^2/(lu^2-9.896161^2))");
    m.component().create("comp1", true);
    m.component("comp1").geom().create("geom1", 2);
    m.component("comp1").geom("geom1").lengthUnit("nm");
    m.component("comp1").geom("geom1").create("air", "Rectangle");
    m.component("comp1").geom("geom1").feature("air").set("pos", new String[]{"-P/2", "0"});
    m.component("comp1").geom("geom1").feature("air").set("size", new String[]{"P", "Hg+B"});
    m.component("comp1").geom("geom1").feature("air").set("selresult", true);
    m.component("comp1").geom("geom1").create("sub", "Rectangle");
    m.component("comp1").geom("geom1").feature("sub").set("pos", new String[]{"-P/2", "-B"});
    m.component("comp1").geom("geom1").feature("sub").set("size", new String[]{"P", "B"});
    m.component("comp1").geom("geom1").feature("sub").set("selresult", true);
    m.component("comp1").geom("geom1").create("ridge", "Polygon");
    m.component("comp1").geom("geom1").feature("ridge").set("source", "vectors");
    m.component("comp1").geom("geom1").feature("ridge").set("x", new String[]{"-P*bottom_duty/2", "P*bottom_duty/2", "P*top_duty/2", "-P*top_duty/2"});
    m.component("comp1").geom("geom1").feature("ridge").set("y", new String[]{"0", "0", "Hg", "Hg"});
    m.component("comp1").geom("geom1").feature("ridge").set("selresult", true);
    m.component("comp1").geom("geom1").run();
    box(m,"top","-P","P","Hg+B-0.01[nm]","Hg+B+0.01[nm]");
    box(m,"bottom","-P","P","-B-0.01[nm]","-B+0.01[nm]");
    box(m,"left","-P/2-0.01[nm]","-P/2+0.01[nm]","-2*B","Hg+2*B");
    box(m,"right","P/2-0.01[nm]","P/2+0.01[nm]","-2*B","Hg+2*B");
    m.component("comp1").selection().create("sides", "Union");
    m.component("comp1").selection("sides").set("entitydim",1);
    m.component("comp1").selection("sides").set("input",new String[]{"left","right"});
    material(m,"airmat","Air",null,"1");
    material(m,"submat","Fused silica substrate","geom1_sub_dom","nSi");
    material(m,"ridgemat","Fused silica ridge (air when height=0)","geom1_ridge_dom","if(height_nm>0,nSi,1)");
    physics(m,"ewfd",true);
    physics(m,"ewfd2",false);
    m.component("comp1").mesh().create("mesh1");
    m.component("comp1").mesh("mesh1").create("size1","Size");
    m.component("comp1").mesh("mesh1").feature("size1").set("custom",true);
    m.component("comp1").mesh("mesh1").feature("size1").set("hmax","450[nm]/(1.48*12*mesh_scale)");
    m.component("comp1").mesh("mesh1").feature("size1").set("hmin","0.5[nm]/mesh_scale");
    m.component("comp1").mesh("mesh1").feature("size1").set("hgrad",1.2);
    m.component("comp1").mesh("mesh1").feature("size1").set("hcurve",0.2);
    m.component("comp1").mesh("mesh1").feature("size1").set("hnarrow",2);
    m.component("comp1").mesh("mesh1").create("edg1","Edge");
    m.component("comp1").mesh("mesh1").feature("edg1").selection().named("left");
    m.component("comp1").mesh("mesh1").create("copy1","CopyEdge");
    m.component("comp1").mesh("mesh1").feature("copy1").selection("source").named("left");
    m.component("comp1").mesh("mesh1").feature("copy1").selection("destination").named("right");
    m.component("comp1").mesh("mesh1").create("tri1","FreeTri");
    m.component("comp1").mesh("mesh1").run();
    study(m,"stdALL","ewfd","ewfd2","INITIAL - 45 paired conditions, TE and TM together");
    m.study("stdALL").feature("freq").activate("ewfd2",true);
    m.study("stdALL").createAutoSequences("all");
    for(String ds:m.result().dataset().tags()) System.out.println("DATASET "+ds+" "+m.result().dataset(ds).getString("solution"));
    String[] datasets=m.result().dataset().tags();
    String data=datasets[datasets.length-1];
    m.result().table().create("tblPowers","Table");
    m.result().table("tblPowers").label("Independent FEM powers: evaluate then export");
    m.result().numerical().create("powers","EvalGlobal");
    m.result().numerical("powers").label("INITIAL RESULTS - all 45 cases, both polarizations");
    m.result().numerical("powers").set("data",data);
    m.result().numerical("powers").set("expr",new String[]{"case_id","period_nm","height_nm","top_duty","bottom_duty","wavelength_nm","angle_deg","comp1.ewfd.Rtotal","comp1.ewfd.Ttotal","comp1.ewfd2.Rtotal","comp1.ewfd2.Ttotal","abs(comp1.ewfd.Rtotal+comp1.ewfd.Ttotal-1)","abs(comp1.ewfd2.Rtotal+comp1.ewfd2.Ttotal-1)"});
    m.result().numerical("powers").set("descr",new String[]{"case_id","period_nm","height_nm","top_duty","bottom_duty","wavelength_nm","angle_deg","R_TE","T_TE","R_TM","T_TM","energy_TE","energy_TM"});
    m.result().numerical("powers").set("table","tblPowers");
    m.result().numerical("powers").set("outerinput","all");
    m.result().export().create("exportPowers","Table");
    m.result().export("exportPowers").label("Export independent FEM powers to comsol_raw.csv");
    m.result().export("exportPowers").set("table","tblPowers");
    m.result().export("exportPowers").set("filename","comsol_raw.csv");
    m.result().export("exportPowers").set("prec","full");
    // Standalone single-condition studies are retained for easy numerical checks.
    singleStudy(m,"probeTE","ewfd","ewfd2","TE - current parameters only");
    singleStudy(m,"probeTM","ewfd2","ewfd","TM - current parameters only");
    System.out.println("MODEL_BUILD_OK paired_cases=45 total_conditions=90");
    return m;
  }
  static void box(Model m,String tag,String xmin,String xmax,String ymin,String ymax) {
    m.component("comp1").selection().create(tag,"Box");
    m.component("comp1").selection(tag).set("entitydim",1);
    m.component("comp1").selection(tag).set("condition","inside");
    m.component("comp1").selection(tag).set("xmin",xmin);
    m.component("comp1").selection(tag).set("xmax",xmax);
    m.component("comp1").selection(tag).set("ymin",ymin);
    m.component("comp1").selection(tag).set("ymax",ymax);
  }
  static void material(Model m,String tag,String label,String sel,String n) {
    m.component("comp1").material().create(tag,"Common");
    m.component("comp1").material(tag).label(label);
    if(sel==null) m.component("comp1").material(tag).selection().all();
    else m.component("comp1").material(tag).selection().named(sel);
    m.component("comp1").material(tag).propertyGroup("def").set("relpermittivity",new String[]{"("+n+")^2","0","0","0","("+n+")^2","0","0","0","("+n+")^2"});
    m.component("comp1").material(tag).propertyGroup("def").set("relpermeability",new String[]{"1","0","0","0","1","0","0","0","1"});
    m.component("comp1").material(tag).propertyGroup("def").set("electricconductivity",new String[]{"0","0","0","0","0","0","0","0","0"});
  }
  static void physics(Model m,String tag,boolean te) {
    m.component("comp1").physics().create(tag,"ElectromagneticWavesFrequencyDomain","geom1");
    m.component("comp1").physics(tag).identifier(tag);
    m.component("comp1").physics(tag).label(te?"TE (out-of-plane Ez)":"TM (in-plane Ex,Ey)");
    m.component("comp1").physics(tag).field("electricfield").field(te?"Ete":"Etm");
    m.component("comp1").physics(tag).prop("components").set("components",te?"outofplane":"inplane");
    for(int i=1;i<=2;i++) {
      String pt="port"+i;
      m.component("comp1").physics(tag).create(pt,"Port",1);
      m.component("comp1").physics(tag).feature(pt).selection().named(i==1?"top":"bottom");
      m.component("comp1").physics(tag).feature(pt).set("PortType","Periodic");
      m.component("comp1").physics(tag).feature(pt).set("PortName",Integer.toString(i));
      m.component("comp1").physics(tag).feature(pt).set("PortExcitation",i==1?"on":"off");
      m.component("comp1").physics(tag).feature(pt).set("InputType",te?"E":"H");
      m.component("comp1").physics(tag).feature(pt).set(te?"Eampl":"Hampl",new String[]{"0","0","1"});
      m.component("comp1").physics(tag).feature(pt).set("n",new String[]{i==1?"1":"nSi","0","0","0",i==1?"1":"nSi","0","0","0",i==1?"1":"nSi"});
      if(i==1) {
        m.component("comp1").physics(tag).feature(pt).set("Pin","1[W/m]");
        m.component("comp1").physics(tag).feature(pt).set("alpha_inc","alpha");
      }
    }
    m.component("comp1").physics(tag).create("floquet","PeriodicCondition",1);
    m.component("comp1").physics(tag).feature("floquet").selection().named("sides");
    m.component("comp1").physics(tag).feature("floquet").set("PeriodicType","Floquet");
    m.component("comp1").physics(tag).feature("floquet").set("Floquet_source","FromPeriodicPort");
  }
  static void study(Model m,String tag,String phys,String other,String label) {
    m.study().create(tag); m.study(tag).label(label);
    m.study(tag).create("param","Parametric");
    m.study(tag).feature("param").set("sweeptype","sparse");
    m.study(tag).feature("param").set("pname",NAMES);
    m.study(tag).feature("param").set("plistarr",LISTS);
    m.study(tag).feature("param").set("punit",new String[]{"","","","","","",""});
    frequency(m,tag,phys,other);
  }
  static void singleStudy(Model m,String tag,String phys,String other,String label) {
    m.study().create(tag); m.study(tag).label(label); frequency(m,tag,phys,other);
  }
  static void frequency(Model m,String tag,String phys,String other) {
    m.study(tag).create("freq","Frequency");
    m.study(tag).feature("freq").set("punit","Hz");
    m.study(tag).feature("freq").set("plist","c_const/lam");
    m.study(tag).feature("freq").activate(phys,true);
    m.study(tag).feature("freq").activate(other,false);
  }
}
