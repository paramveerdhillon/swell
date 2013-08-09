package edu.upenn.cis.SpectralLearning.Runs;

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

import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import Jama.Matrix;
import edu.upenn.cis.SpectralLearning.IO.Options;
import edu.upenn.cis.SpectralLearning.MathUtils.MatrixFormatConversion;
import edu.upenn.cis.SpectralLearning.MathUtils.SVDTemplates;
import edu.upenn.cis.SpectralLearning.SpectralRepresentations.ContextPCARepresentation;

public class CCAVariantsRun implements Serializable {

	private Options _opt;
	private ContextPCARepresentation _cpcaR; 
	private int dim2=0;
	static final long serialVersionUID = 42L;
	Matrix phiL, phiR, phiLT,phiRT;
	double[] s;
	public CCAVariantsRun(Options opt, ContextPCARepresentation cpcaR){
		_opt=opt;
		_cpcaR=cpcaR;
		dim2=2*_opt.contextSizeOneSide*(_opt.vocabSize+1);
		
		
		System.out.println("+++Entering CCA Compute+++");
		computeCCAVariant(_cpcaR);
			
		writeStats();
	}


	private void writeStats() {
		BufferedWriter writer=null;
		double sSum=0;
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream("Output_Files/CCAVariantStats"),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		try {
			
			
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


	private void computeCCAVariant(ContextPCARepresentation _cpcaR2) {
		
		_cpcaR2.computeTrainLRMatrices();
		SparseDoubleMatrix2D View1, View2, View3,View1T, View2T, View3T, v1, v2,v3,v4,v5,v6,v7;
		
		
		
		SVDTemplates svdTC;
		
			svdTC=new SVDTemplates(_opt,dim2);

		System.out.println("+++Generated CCA Matrices+++");
		
		
		
		if(_opt.typeofDecomp.equals("2viewLvsR")){
			View1=_cpcaR2.getLnMatrix();
			View2=_cpcaR2.getRnMatrix();
			View1T=_cpcaR2.getLnTMatrix();
			View2T=_cpcaR2.getRnTMatrix();
			
			computeCCA2(View1,View1T, View2,View2T,svdTC,_cpcaR2,View1.columns(),View2.columns());
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsL")){
			View1=_cpcaR2.getWnMatrix();
			View2=_cpcaR2.getLnMatrix();
			View1T=_cpcaR2.getWnTMatrix();
			View2T=_cpcaR2.getLnTMatrix();
			
			
			computeCCA2(View1,View1T, View2,View2T,svdTC,_cpcaR2,View1.columns(),View2.columns());
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsR")){
			View1=_cpcaR2.getWnMatrix();
			View2=_cpcaR2.getRnMatrix();
			View1T=_cpcaR2.getWnTMatrix();
			View2T=_cpcaR2.getRnTMatrix();
			computeCCA2(View1,View1T, View2,View2T,svdTC,_cpcaR2,View1.columns(),View2.columns());
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsLR")){
			View2=_cpcaR2.concatenateLR(_cpcaR2.getLnMatrix(), _cpcaR2.getRnMatrix());
			View1=_cpcaR2.getWnMatrix();
			View2T=_cpcaR2.concatenateLRT(_cpcaR2.getLnTMatrix(), _cpcaR2.getRnTMatrix());
			View1T=_cpcaR2.getWnTMatrix();
			computeCCA2(View1,View1T, View2,View2T,svdTC,_cpcaR2,View1.columns(),View2.columns());
		}
		
		if(_opt.typeofDecomp.equals("3viewLvsWvsR") || _opt.typeofDecomp.equals("TwoStepLRvsW")){
			View1=_cpcaR2.getLnMatrix();
			View2=_cpcaR2.getWnMatrix();
			View3=_cpcaR2.getRnMatrix();
			View1T=_cpcaR2.getLnTMatrix();
			View2T=_cpcaR2.getWnTMatrix();
			View3T=_cpcaR2.getRnTMatrix();
			if(_opt.typeofDecomp.equals("3viewLvsWvsR"))
				computeCCA3(View1,View1T, View2,View2T,View3,View3T,svdTC,_cpcaR2);
		
			if(_opt.typeofDecomp.equals("TwoStepLRvsW"))
				computeCCATwoStepLRvsW(View1,View1T, View2,View2T,View3,View3T,svdTC,_cpcaR2);
		}
		
		
		
		
		
		
	}
	
	

	private void computeCCA2(SparseDoubleMatrix2D view1,
			SparseDoubleMatrix2D view1t, SparseDoubleMatrix2D view2,
			SparseDoubleMatrix2D view2t, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2,int dim1,int dim2) {
		SparseDoubleMatrix2D xtx=new SparseDoubleMatrix2D(view1t.rows(),view1.columns());
		SparseDoubleMatrix2D yty=new SparseDoubleMatrix2D(view2t.rows(),view2.columns());
		SparseDoubleMatrix2D xty=new SparseDoubleMatrix2D(view1t.rows(),view2.columns());
		SparseDoubleMatrix2D ytx=new SparseDoubleMatrix2D(view2t.rows(),view1.columns());
		SparseDoubleMatrix2D auxMat1=new SparseDoubleMatrix2D(xtx.rows(),xty.columns());
		SparseDoubleMatrix2D auxMat2=new SparseDoubleMatrix2D(yty.rows(),ytx.columns());
		SparseDoubleMatrix2D auxMat3=new SparseDoubleMatrix2D(auxMat1.rows(),auxMat1.rows());
		SparseDoubleMatrix2D auxMat4=new SparseDoubleMatrix2D(auxMat2.rows(),auxMat2.rows());
				
				
		view1t.zMult(view1, xtx);
		view2t.zMult(view2, yty);
		view1t.zMult(view2, xty);
		view2t.zMult(view1, ytx);
		
		(svdTC.computeSparseInverse(xtx)).zMult(xty, auxMat1);
		(svdTC.computeSparseInverse(yty)).zMult(ytx, auxMat2);
		auxMat1.zMult(auxMat2,auxMat3);
		phiL=svdTC.computeSVD_Tropp(auxMat3, _cpcaR2.getOmegaMatrix(auxMat3.columns()),dim1);
		s=svdTC.getSingularVals();
		
		auxMat2.zMult(auxMat1,auxMat4);
		phiR=svdTC.computeSVD_Tropp(auxMat4, _cpcaR2.getOmegaMatrix(auxMat4.columns()),dim2);
		
	}
	
	private void computeCCATwoStepLRvsW(SparseDoubleMatrix2D view1,
			SparseDoubleMatrix2D view1t, SparseDoubleMatrix2D view2,
			SparseDoubleMatrix2D view2t, SparseDoubleMatrix2D view3,
			SparseDoubleMatrix2D view3t, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2) {
		
		SparseDoubleMatrix2D LphiL=new SparseDoubleMatrix2D(view1.rows(),_opt.hiddenStateSize);
		SparseDoubleMatrix2D RphiR=new SparseDoubleMatrix2D(view3.rows(),_opt.hiddenStateSize);
		SparseDoubleMatrix2D LTphiL=new SparseDoubleMatrix2D(_opt.hiddenStateSize,view1.rows());
		SparseDoubleMatrix2D RTphiR=new SparseDoubleMatrix2D(_opt.hiddenStateSize,view3.rows());
		SparseDoubleMatrix2D LphiLRphiR=new SparseDoubleMatrix2D(view1.rows(),view1.columns()+view3.columns());
		SparseDoubleMatrix2D LphiLRphiRT=new SparseDoubleMatrix2D(view1.columns()+view3.columns(),view1.rows());
				
		
		
		
		computeCCA2(view1,view1t, view3,view3t,svdTC,_cpcaR2,view1.columns(),view3.columns());
		
		view1.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), LphiL);
		view3.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), RphiR);
		
		phiLT=phiL.transpose();
		phiRT=phiR.transpose();
		
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(view1t, LTphiL);
		MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(view3t, RTphiR);
		
		LphiLRphiR=_cpcaR2.concatenateLR(LphiL,RphiR);
		LphiLRphiRT=_cpcaR2.concatenateLRT(LTphiL,RTphiR);
		
		computeCCA2(LphiLRphiR,LphiLRphiRT, view2,view2t,svdTC,_cpcaR2,LphiLRphiR.columns(),view2.columns());
		
	}



	private void computeCCA3(SparseDoubleMatrix2D view1,
			SparseDoubleMatrix2D view1t, SparseDoubleMatrix2D view2,
			SparseDoubleMatrix2D view2t, SparseDoubleMatrix2D view3,
			SparseDoubleMatrix2D view3t, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2) {
		// TODO Auto-generated method stub
		
	}


	

	public Matrix getPhiL(){
		return phiL;
	}
	
	public Matrix getPhiR(){
		return phiR;
	}
	
	
public void serializeCCAVariantsRun() {
		
		
		String contextDict=_opt.serializeRun+"Context";
		File fContext= new File(contextDict);
		
		String eigDict=_opt.serializeRun+"Eig";
		File fEig= new File(eigDict);
			
		try{
			ObjectOutput ccaEig=new ObjectOutputStream(new FileOutputStream(fEig));
			ObjectOutput ccaContext=new ObjectOutputStream(new FileOutputStream(fContext));
			
		if(!_opt.typeofDecomp.equals("TwoStepLRvsW")){
				ccaEig.writeObject(phiL);
				ccaEig.flush();
				ccaEig.close();
			
				ccaContext.writeObject(phiR);
				ccaContext.flush();
				ccaContext.close();
		}else{
			ccaEig.writeObject(phiR);
			ccaEig.flush();
			ccaEig.close();
		
			ccaContext.writeObject(phiL);
			ccaContext.flush();
			ccaContext.close();
		}
			
			System.out.println("=======Serialized the CCA Variant Run=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
	
}
}
