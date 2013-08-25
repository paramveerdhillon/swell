package edu.upenn.cis.swell.Runs;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

import edu.upenn.cis.swell.IO.Options;
import edu.upenn.cis.swell.MathUtils.CenterScaleNormalizeUtils;
import edu.upenn.cis.swell.SpectralRepresentations.CCARepresentation;
import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class CCARun implements Serializable {
	
	private int _num_hidden;
	private Options _opt;
	private ArrayList<Double> _smooths =new ArrayList<Double>();
	private int _rank;
	private SingularValueDecomposition _svd;
	private Matrix eigenFeatDictL,eigenFeatDictLAllk;
	private Matrix eigenFeatDict;
	private Matrix normSingVal;
	private Matrix sVals;
	private Matrix eigenFeatDictR,eigenFeatDictRAllk;
	private CCARepresentation _ccaR;
	private double[] s=new double[_num_hidden];
	private double prevNorm=0;
	private double currNorm=0;
	private CenterScaleNormalizeUtils mathUtils=null;
	static final long serialVersionUID = 42L;
	
	public CCARun(Options opt, CCARepresentation ccaR){
		_opt=opt;
		_ccaR=ccaR;
		_num_hidden=opt.hiddenStateSize;
		_smooths=opt.smoothArray;
		mathUtils=new CenterScaleNormalizeUtils(_opt);

		for(int i=0;i<_opt.numIters;i++){
			final long startTime = System.currentTimeMillis();	
			computeCCA(_ccaR);
			prevNorm=currNorm;
			System.out.println("===Computed CCA===");
			updateEigenDict();
			System.out.println("===Updated dict.===");
			currNorm=computeMatrixNorm();
			System.out.println("===Iteration Number: "+(i+1)+" Finished===");
			final long endTime = System.currentTimeMillis();
			System.out.println("===Time taken for Iteration: "+(endTime-startTime)/1000.0/60.0+" mins===");
			
			
			printConvergenceStats();
		}
		writeStats();
	}
	
	
	private void computeCCA(CCARepresentation ccaRep){
		
		
		
		ccaRep.generateCovForAllDocs();
		
		Matrix ccaM= ccaRep.getCovLLAllDocsMatrix().inverse().times(ccaRep.getCovLRAllDocsMatrix()).times(ccaRep.getCovRRAllDocsMatrix().inverse()).times(ccaRep.getCovLRAllDocsMatrix().transpose());
		_svd=ccaM.svd();
		
		
		normSingVal=_svd.getS().times(1/_svd.getS().normF());
		
		//System.out.println(normSingVal.normF());
		//System.out.println(normSingVal.norm1());
		//System.out.println(_svd.getS().normF());
		
		if(_opt.scaleBySingVals){
			eigenFeatDictL=_svd.getU().times(normSingVal).getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden/2)-1);
			eigenFeatDictLAllk=_svd.getU().times(normSingVal).getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden)-1);
			sVals=_svd.getS().getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden)-1);
			eigenFeatDictR=_svd.getV().times(normSingVal).getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden/2)-1);
			eigenFeatDictRAllk=_svd.getV().times(normSingVal).getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden)-1);
			_rank=_svd.rank();
			s=_svd.getSingularValues();
		}
		else{
			eigenFeatDictL=_svd.getU().getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden/2)-1);
			eigenFeatDictLAllk=_svd.getU().getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden)-1);
			sVals=_svd.getS().getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden)-1);
			eigenFeatDictR=_svd.getV().getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden/2)-1);
			eigenFeatDictRAllk=_svd.getV().getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden)-1);
			_rank=_svd.rank();
			s=_svd.getSingularValues();
			
		}
		
		
	}
	
	public int getRank(){
		return _rank;
	}
	
	public Matrix getSingularValues(){
		return sVals;
	}
	
	public Matrix getLeftSingularVecs(){
		return eigenFeatDictL;
	}
	
	public Matrix getLeftSingularVecsAllk(){
		return eigenFeatDictLAllk;
	}
	
	public Matrix getRightSingularVecsAllk(){
		return eigenFeatDictRAllk;
	}
	
	public Matrix getEigenFeatDict(){
		eigenFeatDict=(Matrix)_ccaR.getEigenFeatDict().clone();
		return eigenFeatDict;
	}
	
	public Matrix getRightSingularVecs(){
		return eigenFeatDictR;
	}
	
	public SingularValueDecomposition getSVD(){
		return _svd;
	}
	
	
	
	public void serializeCCARun(){
		String leftEig=_opt.serializeRun+"L";
		File fL= new File(leftEig);
		
		String rightEig=_opt.serializeRun+"R";
		File fR= new File(rightEig);
		
		String eigDict=_opt.serializeRun+"Eig";
		File fEig= new File(eigDict);
		try{
			ObjectOutput ccaL=new ObjectOutputStream(new FileOutputStream(fL));
			ObjectOutput ccaR=new ObjectOutputStream(new FileOutputStream(fR));
			ObjectOutput ccaEig=new ObjectOutputStream(new FileOutputStream(fEig));
			ccaL.writeObject((Object)eigenFeatDictLAllk);
			ccaL.flush();
			ccaL.close();
			ccaR.writeObject((Object)eigenFeatDictRAllk);
			ccaR.flush();
			ccaR.close();
			ccaEig.writeObject((Object)getEigenFeatDict());
			ccaEig.flush();
			ccaEig.close();
			
			System.out.println("=======Serialized the CCA Run=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
	}
	
	
	public void updateEigenDict(){
		
		Matrix neweigenDict=null;
		Matrix eigenDict=_ccaR.getEigenFeatDict();
		
		_ccaR.left_right_smooths_W(eigenDict,eigenFeatDictL,eigenFeatDictR);
		neweigenDict=mathUtils.center_and_scale(_ccaR.getEigenFeatDict());
		
		
		_ccaR.setEigenFeatDict(neweigenDict);		
	}
	
	
	private void writeStats() {
		BufferedWriter writer=null;
		double sSum=0;
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream("Output_Files/Stats"),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		try {
			
			
			writer.write("Rank after SVD: "+_rank+"\n\n");
			writer.write("Eigenvalues in decreasing order:\n");
			for (int i=0;i<s.length;i++){
				Double d=new Double(s[i]);
				writer.write(d.toString()+"\n");
				sSum+=d.doubleValue();
				
			}
			writer.write("\n\nNormalized Eigenvalues in decreasing order:\n");
			for (int i=0;i<s.length;i++){
				Double d=new Double(s[i]/sSum);
				writer.write(d.toString()+"\n");
				
			}
			
			writer.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		
	}
	
	
	public double computeMatrixNorm(){
		return _ccaR.getEigenFeatDict().normF();
	}
	
	public void printConvergenceStats(){
		System.out.println("====Absolute Change in EigenFeatDict Frob. Norm: "+Math.abs(currNorm-prevNorm)+"=====\n");
		
	}

}
