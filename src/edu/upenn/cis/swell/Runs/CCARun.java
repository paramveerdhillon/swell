package edu.upenn.cis.swell.Runs;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */


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
import edu.upenn.cis.swell.MathUtils.MatrixFormatConversion;
import edu.upenn.cis.swell.MathUtils.SVDTemplates;
import edu.upenn.cis.swell.SpectralRepresentations.CCARepresentation;
import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class CCARun implements Serializable {
	
	private int _num_hidden;
	private Options _opt;
	private ArrayList<Double> _smooths =new ArrayList<Double>();
	//private int _rank;
	private EigenvalueDecomposition _eigL,_eigR;
	private Matrix eigenFeatDictL,eigenFeatDictLAllk;
	private Matrix eigenFeatDict;
	//private Matrix normEigValL,normEigValR;
	private double[] realeigValsL,compeigValsL,realeigValsR,compeigValsR;
	private Matrix eigenFeatDictR,eigenFeatDictRAllk;
	private CCARepresentation _ccaR;
	//private double[] s=new double[_num_hidden];
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
		//Matrix neweigenDict=mathUtils.center_and_scale(_ccaR.getEigenFeatDict());
		//_ccaR.setEigenFeatDict(neweigenDict);		
		
		
		writeStats();
	}
	
	
	private void computeCCA(CCARepresentation ccaRep){
		
		SVDTemplates svdTC;
		svdTC=new SVDTemplates(_opt,ccaRep.getCovLLAllDocsMatrix().getColumnDimension());
		
		ccaRep.generateCovForAllDocs();
		
		Matrix ccaM= ccaRep.getCovLLAllDocsMatrix().inverse().times(ccaRep.getCovLRAllDocsMatrix()).times(ccaRep.getCovRRAllDocsMatrix().inverse()).times(ccaRep.getCovRLAllDocsMatrix());
		_eigL=ccaM.eig();
		
		ccaM= ccaRep.getCovRRAllDocsMatrix().inverse().times(ccaRep.getCovRLAllDocsMatrix()).times(ccaRep.getCovLLAllDocsMatrix().inverse()).times(ccaRep.getCovLRAllDocsMatrix());
		_eigR=ccaM.eig();
		
		/*
		 * 
		 * Dont normalize by eigenvalues since they can be negative
		normEigValL=_eigL.getD();
		
		normEigValR=_eigR.getD();
		double ssL =0,ssR=0;
		for(int j=0; j<normEigValL.getRowDimension();j++){
			
			ssL += Math.abs(normEigValL.get(j,j));
			ssR += Math.abs(normEigValL.get(j,j));	
		}
		for(int j=0; j<normEigValL.getRowDimension();j++){
			
			normEigValL.set(j,j, Math.abs(normEigValL.get(j,j)/ssL));
			normEigValR.set(j,j, Math.abs(normEigValR.get(j,j)/ssR));
		}
		
		*/
		

		
		//System.out.println(normSingVal.normF());
		//System.out.println(normSingVal.norm1());
		//System.out.println(_svd.getS().normF());
		
		/*
		if(_opt.scaleBySingVals){
			eigenFeatDictL=_eigL.getV().times(normEigValL).getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden/2)-1);
			eigenFeatDictLAllk=_eigL.getV().times(normEigValL).getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden)-1);
			eigValsL=_eigL.getD().getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden)-1);
			eigenFeatDictR=_eigR.getV().times(normEigValR).getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden/2)-1);
			eigenFeatDictRAllk=_eigR.getV().times(normEigValR).getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden)-1);
			eigValsR=_eigR.getD().getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden)-1);
			
		}
		else{
		*/
			eigenFeatDictL=MatrixFormatConversion.createDenseMatrixJAMA(svdTC.computeDenseInverseSqRoot(MatrixFormatConversion.createDenseMatrixCOLT(ccaRep.getCovLLAllDocsMatrix()))).times(_eigL.getV().getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden/2)-1));
			eigenFeatDictLAllk=_eigL.getV().getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden)-1);
			realeigValsL=_eigL.getRealEigenvalues();
			compeigValsL=_eigL.getImagEigenvalues();
			eigenFeatDictR=MatrixFormatConversion.createDenseMatrixJAMA(svdTC.computeDenseInverseSqRoot(MatrixFormatConversion.createDenseMatrixCOLT(ccaRep.getCovRRAllDocsMatrix()))).times(_eigR.getV().getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden/2)-1));
			eigenFeatDictRAllk=_eigR.getV().getMatrix(0,(_smooths.size()*_num_hidden)-1,0,(_num_hidden)-1);
			realeigValsR=_eigR.getRealEigenvalues();
			compeigValsR=_eigR.getImagEigenvalues();
			
		//}
		
		
	}
	
	
	public Matrix scaleEigenvectors(Matrix eigVecs, double[] rEvals, double[] cEvals){
		
		Matrix eigVecsNew=eigVecs;
		Double[] d1=new Double[rEvals.length];
		
		double sSum=0,s=0;
		
		for (int i=0;i<rEvals.length;i++){
			s =Math.sqrt((rEvals[i]*rEvals[i]) + (cEvals[i]*cEvals[i])) ;
			Double d=new Double(s);
			sSum+=d.doubleValue();
			
		}
		for (int i=0;i<rEvals.length;i++){
			s =Math.sqrt((rEvals[i]*rEvals[i]) + (cEvals[i]*cEvals[i])) ;
			d1[i]=s/sSum;
		}
		
		for(int ii =0; ii<eigVecs.getRowDimension();ii++){
			for(int jj =0; jj<eigVecs.getColumnDimension();jj++){
				eigVecsNew.set(ii, jj, eigVecs.get(ii, jj)*d1[jj]);
			}
		}
		
		return eigVecsNew;
	}
	
	//public Matrix getEigenValuesL(){
	//	return eigValsL;
	//}
	
	//public Matrix getEigenValuesR(){
	//	return eigValsR;
	//}
	
	public Matrix getLeftEigenVecs(){
		
		//return scaleEigenvectors(eigenFeatDictL,realeigValsL,compeigValsL);
		
		return eigenFeatDictL;
	}
	public Matrix getRightEigenVecs(){
		//return scaleEigenvectors(eigenFeatDictR,realeigValsR,compeigValsR);
		
		return eigenFeatDictR;
	}
	
	
	public Matrix getLeftEigenVecsAllk(){
		return eigenFeatDictLAllk;
	}
	
	public Matrix getRightEigenVecsAllk(){
		return eigenFeatDictRAllk;
	}
	
	public Matrix getEigenFeatDict(){
		eigenFeatDict=(Matrix)_ccaR.getEigenFeatDict().clone();
		return eigenFeatDict;
	}
	
	
	
	public EigenvalueDecomposition getEigL(){
		return _eigL;
	}
	
	public EigenvalueDecomposition getEigR(){
		return _eigR;
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
		
		_ccaR.left_right_smooths_W(eigenDict,getLeftEigenVecs(),getRightEigenVecs());
		//neweigenDict=mathUtils.center_and_scale(_ccaR.getEigenFeatDict());
		
		neweigenDict=mathUtils.normalize(_ccaR.getEigenFeatDict());
		
		//neweigenDict=mathUtils.center(_ccaR.getEigenFeatDict());
		
		_ccaR.setEigenFeatDict(neweigenDict);		
	}
	
	
	private void writeStats() {
		BufferedWriter writer=null;
		double sSum=0;
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.outputDir+"Stats"),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		try {
			
			double s;
			writer.write("Left Eigenvalues in decreasing order:\n");
			for (int i=0;i<_num_hidden;i++){
				s =Math.sqrt((realeigValsL[i]*realeigValsL[i]) + (compeigValsL[i]*compeigValsL[i])) ;
				Double d=new Double(s);
				writer.write(d.toString()+"\n");
				sSum+=d.doubleValue();
				
			}
			writer.write("\n\nNormalized Left Eigenvalues in decreasing order:\n");
			for (int i=0;i<_num_hidden;i++){
				s =Math.sqrt((realeigValsL[i]*realeigValsL[i]) + (compeigValsL[i]*compeigValsL[i])) ;
				Double d=new Double(s/sSum);
				writer.write(d.toString()+"\n");
				
			}
			///////////////
			sSum=0;
			writer.write("\n\nRight Eigenvalues in decreasing order:\n");
			for (int i=0;i<_num_hidden;i++){
				s =Math.sqrt((realeigValsR[i]*realeigValsR[i]) + (compeigValsR[i]*compeigValsR[i])) ;
				Double d=new Double(s);
				writer.write(d.toString()+"\n");
				sSum+=d.doubleValue();
				
			}
			writer.write("\n\nNormalized Right Eigenvalues in decreasing order:\n");
			for (int i=0;i<_num_hidden;i++){
				s =Math.sqrt((realeigValsR[i]*realeigValsR[i]) + (compeigValsR[i]*compeigValsR[i])) ;
				Double d=new Double(s/sSum);
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
