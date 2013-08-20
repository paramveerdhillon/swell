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
		
		_cpcaR2.computeContextLRMatrices();
		SparseDoubleMatrix2D wtl,ltw,wtw,ltl,rtr,wtr,rtw,wtlr,lrtw,ltr,rtl,lrtlr;
		
		
		
		SVDTemplates svdTC;
		
			svdTC=new SVDTemplates(_opt,dim2);

		System.out.println("+++Generated CCA Matrices+++");
		
		
		
		if(_opt.typeofDecomp.equals("2viewLvsR")){
			ltr=_cpcaR2.getLTRMatrix();
			rtl=_cpcaR2.getRTLMatrix();
			ltl=_cpcaR2.getLTLMatrix();
			rtr=_cpcaR2.getRTRMatrix();
			
			computeCCA2(ltr,rtl,ltl,rtr,svdTC,_cpcaR2,ltr.rows(),ltr.columns());
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsL")){
			wtl=_cpcaR2.getWTLMatrix();
			ltw=_cpcaR2.getLTWMatrix();
			ltl=_cpcaR2.getLTLMatrix();
			wtw=_cpcaR2.getWTWMatrix();
			
			
			computeCCA2(wtl,ltw,ltl,wtw,svdTC,_cpcaR2,wtl.rows(),wtl.columns());
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsR")){
			wtr=_cpcaR2.getWTRMatrix();
			rtw=_cpcaR2.getRTWMatrix();
			rtr=_cpcaR2.getRTRMatrix();
			wtw=_cpcaR2.getWTWMatrix();
			computeCCA2(wtr,rtw,rtr,wtw,svdTC,_cpcaR2,wtr.rows(),wtr.columns());
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsLR")){
			wtlr=_cpcaR2.getWTLRMatrix();
			lrtw=_cpcaR2.getLRTWMatrix();
			lrtlr=_cpcaR2.getLRTLRMatrix();
			wtw=_cpcaR2.getWTWMatrix();
			computeCCA2(wtlr, lrtw,lrtlr,wtw,svdTC,_cpcaR2,wtlr.rows(),wtlr.columns());
		}
		
		if(_opt.typeofDecomp.equals("3viewLvsWvsR") || _opt.typeofDecomp.equals("TwoStepLRvsW")){
			ltr=_cpcaR2.getLTRMatrix();
			rtl=_cpcaR2.getRTLMatrix();
			ltl=_cpcaR2.getLTLMatrix();
			rtr=_cpcaR2.getRTRMatrix();
			wtw=_cpcaR2.getWTWMatrix();
			wtl=_cpcaR2.getWTLMatrix();
			wtr=_cpcaR2.getWTRMatrix();
			ltw=_cpcaR2.getLTWMatrix();
			rtw=_cpcaR2.getRTWMatrix();
			
			//if(_opt.typeofDecomp.equals("3viewLvsWvsR"))
				//computeCCA3(View1,View1T, View2,View2T,View3,View3T,svdTC,_cpcaR2);
		
			if(_opt.typeofDecomp.equals("TwoStepLRvsW"))
				computeCCATwoStepLRvsW(ltr,rtl,ltl,rtr,wtw,wtl,wtr,ltw,rtw,svdTC,_cpcaR2);
		}
		
		
		
		
		
		
	}
	
	

	private void computeCCA2(SparseDoubleMatrix2D xty,
			SparseDoubleMatrix2D ytx, SparseDoubleMatrix2D yty,
			SparseDoubleMatrix2D xtx, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2,int dim1,int dim2) {
		
		SparseDoubleMatrix2D auxMat1=new SparseDoubleMatrix2D(xtx.rows(),xty.columns());
		SparseDoubleMatrix2D auxMat2=new SparseDoubleMatrix2D(yty.rows(),ytx.columns());
		SparseDoubleMatrix2D auxMat3=new SparseDoubleMatrix2D(auxMat1.rows(),auxMat1.rows());
		SparseDoubleMatrix2D auxMat4=new SparseDoubleMatrix2D(auxMat2.rows(),auxMat2.rows());
				
				
		(svdTC.computeSparseInverse(xtx)).zMult(xty, auxMat1);
		(svdTC.computeSparseInverse(yty)).zMult(ytx, auxMat2);
		auxMat1.zMult(auxMat2,auxMat3);
		phiL=svdTC.computeSVD_Tropp(auxMat3, _cpcaR2.getOmegaMatrix(auxMat3.columns()),dim1);
		s=svdTC.getSingularVals();
		
		auxMat2.zMult(auxMat1,auxMat4);
		phiR=svdTC.computeSVD_Tropp(auxMat4, _cpcaR2.getOmegaMatrix(auxMat4.columns()),dim2);
		
	}
	
	private void computeCCATwoStepLRvsW(SparseDoubleMatrix2D ltr,
			SparseDoubleMatrix2D rtl, SparseDoubleMatrix2D ltl,
			SparseDoubleMatrix2D rtr, SparseDoubleMatrix2D wtw,
			SparseDoubleMatrix2D wtl, SparseDoubleMatrix2D wtr, SparseDoubleMatrix2D ltw, SparseDoubleMatrix2D rtw, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2) {
		
		
		
		SparseDoubleMatrix2D WTLphiL=new SparseDoubleMatrix2D(wtl.rows(),_opt.hiddenStateSize);
		SparseDoubleMatrix2D WTRphiR=new SparseDoubleMatrix2D(wtr.rows(),_opt.hiddenStateSize);
		SparseDoubleMatrix2D LTWphiL=new SparseDoubleMatrix2D(_opt.hiddenStateSize,wtl.rows());
		SparseDoubleMatrix2D RTWphiR=new SparseDoubleMatrix2D(_opt.hiddenStateSize,wtr.rows());
		SparseDoubleMatrix2D WTLphiLWTRphiR=new SparseDoubleMatrix2D(wtl.rows(),2*_opt.hiddenStateSize);
		SparseDoubleMatrix2D phiLTLTWphiRTRTW=new SparseDoubleMatrix2D(2*_opt.hiddenStateSize,wtl.rows());
		
		SparseDoubleMatrix2D LRTLRphiLphiR=new SparseDoubleMatrix2D(2*_opt.hiddenStateSize,2*_opt.hiddenStateSize);
				
		SparseDoubleMatrix2D phiLT_LTL_phiL=new SparseDoubleMatrix2D(2*_opt.hiddenStateSize,2*_opt.hiddenStateSize);
		SparseDoubleMatrix2D phiLT_LTR_phiR=new SparseDoubleMatrix2D(2*_opt.hiddenStateSize,2*_opt.hiddenStateSize);
		SparseDoubleMatrix2D phiRT_RTL_phiL=new SparseDoubleMatrix2D(2*_opt.hiddenStateSize,2*_opt.hiddenStateSize);
		SparseDoubleMatrix2D phiRT_RTR_phiR=new SparseDoubleMatrix2D(2*_opt.hiddenStateSize,2*_opt.hiddenStateSize);
		
		SparseDoubleMatrix2D  ltl_phiL=new SparseDoubleMatrix2D(ltl.rows(),_opt.hiddenStateSize);
		SparseDoubleMatrix2D  ltr_phiR=new SparseDoubleMatrix2D(ltr.rows(),_opt.hiddenStateSize);
		SparseDoubleMatrix2D  rtl_phiL=new SparseDoubleMatrix2D(rtl.rows(),_opt.hiddenStateSize);
		SparseDoubleMatrix2D  rtr_phiR=new SparseDoubleMatrix2D(rtr.rows(),_opt.hiddenStateSize);
		
		
		computeCCA2(ltr,rtl, rtr,ltl,svdTC,_cpcaR2,ltr.rows(),ltr.columns());
		
		wtl.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), WTLphiL);
		wtr.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), WTRphiR);
		
		phiLT=phiL.transpose();
		phiRT=phiR.transpose();
		
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(ltw, LTWphiL);
		MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(rtw, RTWphiR);
		
		WTLphiLWTRphiR=_cpcaR2.concatenateLR(WTLphiL,WTRphiR);
		phiLTLTWphiRTRTW=_cpcaR2.concatenateLR(LTWphiL,RTWphiR);
		
		ltl.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), ltl_phiL);
		ltr.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), ltr_phiR);
		rtl.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), rtl_phiL);
		rtr.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), rtr_phiR);
		
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(ltl_phiL, phiLT_LTL_phiL);
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(ltr_phiR, phiLT_LTR_phiR);
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(rtl_phiL, phiRT_RTL_phiL);
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(rtr_phiR, phiRT_RTR_phiR);
		
		LRTLRphiLphiR=_cpcaR2.concatenateLRT(_cpcaR2.concatenateLR(phiLT_LTL_phiL,phiLT_LTR_phiR),_cpcaR2.concatenateLR(phiRT_RTL_phiL,phiRT_RTR_phiR));
		
		computeCCA2(WTLphiLWTRphiR,phiLTLTWphiRTRTW, LRTLRphiLphiR,wtw,svdTC,_cpcaR2,WTLphiLWTRphiR.rows(),WTLphiLWTRphiR.columns());
		
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
