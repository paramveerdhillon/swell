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

import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import Jama.Matrix;
import edu.upenn.cis.swell.IO.Options;
import edu.upenn.cis.swell.MathUtils.MatrixFormatConversion;
import edu.upenn.cis.swell.MathUtils.SVDTemplates;
import edu.upenn.cis.swell.SpectralRepresentations.ContextPCARepresentation;

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
		if(_opt.kdimDecomp){
			computeCCAVariantDense(_cpcaR);
		}
		else{
			computeCCAVariant(_cpcaR);
		}
			
		writeStats();
	}


	private void writeStats() {
		BufferedWriter writer=null;
		double sSum=0;
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.outputDir+"CCAVariantStats"),"UTF8"));
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
		
		SVDTemplates svdTC;
		
			svdTC=new SVDTemplates(_opt,dim2);

		System.out.println("+++Generated CCA Matrices+++");
		
		if(_opt.typeofDecomp.equals("2viewWvsL")){
			
			computeCCA2(MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getWTLMatrix()),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getLTWMatrix()),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getLTLMatrix()),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getWTWMatrix()),svdTC,_cpcaR2);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsR")){
			computeCCA2(MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getWTRMatrix()),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getRTWMatrix()),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getRTRMatrix()),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getWTWMatrix()),svdTC,_cpcaR2);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsLR")){
			computeCCA2(MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getWTLRMatrix()), MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getLRTWMatrix()),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getLRTLRMatrix()),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getWTWMatrix()),svdTC,_cpcaR2);
		}
		
		//3 view is not implemented as yet.
		if(_opt.typeofDecomp.equals("3viewLvsWvsR") || _opt.typeofDecomp.equals("TwoStepLRvsW")){
		
			if(_opt.typeofDecomp.equals("TwoStepLRvsW"))
				computeCCATwoStepLRvsW(MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getLTRMatrix()),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getRTLMatrix()),
						MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getLTLMatrix()),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getRTRMatrix()),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getWTWMatrix()),
						MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getWTLMatrix()),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getWTRMatrix()),
						MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getLTWMatrix()),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getRTWMatrix()),svdTC,_cpcaR2);
		}
	}
	
	private void computeCCAVariantDense(ContextPCARepresentation _cpcaR2) {
		_cpcaR2.computeContextLRDenseMatrices();	
		
		SVDTemplates svdTC;
		
			svdTC=new SVDTemplates(_opt,dim2);

		System.out.println("+++Generated Dense CCA Matrices+++");
		
		if(_opt.typeofDecomp.equals("2viewWvsL")){
			
			computeCCA2(_cpcaR2.getWTLDenseMatrix(),_cpcaR2.getLTWDenseMatrix(),_cpcaR2.getLTLDenseMatrix(),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getWTWMatrix()),svdTC,_cpcaR2);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsR")){
			computeCCA2(_cpcaR2.getWTRDenseMatrix(),_cpcaR2.getRTWDenseMatrix(),_cpcaR2.getRTRDenseMatrix(),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getWTWMatrix()),svdTC,_cpcaR2);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsLR")){
			computeCCA2(_cpcaR2.getWTLRDenseMatrix(),_cpcaR2.getLRTWDenseMatrix(),_cpcaR2.getLRTLRDenseMatrix(),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getWTWMatrix()),svdTC,_cpcaR2);
		}
		
		//3 view is not implemented as yet.
		if(_opt.typeofDecomp.equals("3viewLvsWvsR") || _opt.typeofDecomp.equals("TwoStepLRvsW")){
		
			if(_opt.typeofDecomp.equals("TwoStepLRvsW"))
				computeCCATwoStepLRvsW(_cpcaR2.getLTRDenseMatrix(),_cpcaR2.getRTLDenseMatrix(),
						_cpcaR2.getLTLDenseMatrix(),_cpcaR2.getRTRDenseMatrix(),MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR2.getWTWMatrix()),
						_cpcaR2.getWTLDenseMatrix(),_cpcaR2.getWTRDenseMatrix(),_cpcaR2.getLTWDenseMatrix(),_cpcaR2.getRTWDenseMatrix(),svdTC,_cpcaR2);
		}

		
	}
	
	
	
	private void computeCCA2(DenseDoubleMatrix2D xty,
			DenseDoubleMatrix2D ytx, DenseDoubleMatrix2D yty,
			SparseDoubleMatrix2D xtx, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2) {
		
		
		System.out.println("+++Entering CCA Compute Function+++");
		
		DenseDoubleMatrix2D auxMat1=new DenseDoubleMatrix2D(xtx.rows(),xty.columns());
		DenseDoubleMatrix2D auxMat2=new DenseDoubleMatrix2D(yty.rows(),ytx.columns());
		DenseDoubleMatrix2D auxMat3=new DenseDoubleMatrix2D(auxMat1.rows(),auxMat1.rows());
		DenseDoubleMatrix2D auxMat4=new DenseDoubleMatrix2D(auxMat2.rows(),auxMat2.rows());
				
		int dim1=xty.rows();
		int dim2=xty.columns();
		
		System.out.println("+++Initialized auxiliary matrices+++");
				
		(svdTC.computeSparseInverse(xtx)).zMult(xty, auxMat1);
		
		System.out.println("+++Computed 1 inverse+++");
		
		DenseDoubleAlgebra dalg=new DenseDoubleAlgebra();
		
		((DenseDoubleMatrix2D) dalg.inverse(yty)).zMult(ytx, auxMat2);
		
		System.out.println("+++Computed Inverses+++");
		auxMat1.zMult(auxMat2,auxMat3);
		
		System.out.println("+++Entering SVD computation+++");
		phiL=svdTC.computeSVD_Tropp(auxMat3, _cpcaR2.getOmegaMatrix(auxMat3.columns()),dim1);
		s=svdTC.getSingularVals();
		
		auxMat2.zMult(auxMat1,auxMat4);
		phiR=svdTC.computeSVD_Tropp(auxMat4, _cpcaR2.getOmegaMatrix(auxMat4.columns()),dim2);
		
	}
	
	private void computeCCA2(DenseDoubleMatrix2D xty,
			DenseDoubleMatrix2D ytx, DenseDoubleMatrix2D yty,
			DenseDoubleMatrix2D xtx, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2) {
		
		
		System.out.println("+++Entering CCA Compute Function+++");
		
		DenseDoubleMatrix2D auxMat1=new DenseDoubleMatrix2D(xtx.rows(),xty.columns());
		DenseDoubleMatrix2D auxMat2=new DenseDoubleMatrix2D(yty.rows(),ytx.columns());
		DenseDoubleMatrix2D auxMat3=new DenseDoubleMatrix2D(auxMat1.rows(),auxMat1.rows());
		DenseDoubleMatrix2D auxMat4=new DenseDoubleMatrix2D(auxMat2.rows(),auxMat2.rows());
				
		int dim1=xty.rows();
		int dim2=xty.columns();
		
		DenseDoubleAlgebra dalg=new DenseDoubleAlgebra();
		
		System.out.println("+++Initialized auxiliary matrices+++");
				
		((DenseDoubleMatrix2D) dalg.inverse(xtx)).zMult(xty, auxMat1);
		
		System.out.println("+++Computed 1 inverse+++");
		
		
		
		((DenseDoubleMatrix2D) dalg.inverse(yty)).zMult(ytx, auxMat2);
		
		System.out.println("+++Computed Inverses+++");
		auxMat1.zMult(auxMat2,auxMat3);
		
		System.out.println("+++Entering SVD computation+++");
		phiL=svdTC.computeSVD_Tropp(auxMat3, _cpcaR2.getOmegaMatrix(auxMat3.columns()),dim1);
		s=svdTC.getSingularVals();
		
		auxMat2.zMult(auxMat1,auxMat4);
		phiR=svdTC.computeSVD_Tropp(auxMat4, _cpcaR2.getOmegaMatrix(auxMat4.columns()),dim2);
		
	}
	
	
	

	private void computeCCA2(SparseDoubleMatrix2D xty,
			SparseDoubleMatrix2D ytx, SparseDoubleMatrix2D yty,
			SparseDoubleMatrix2D xtx, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2) {
		
		
		System.out.println("+++Entering CCA Compute Function+++");
		
		SparseDoubleMatrix2D auxMat1=new SparseDoubleMatrix2D(xtx.rows(),xty.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat2=new SparseDoubleMatrix2D(yty.rows(),ytx.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat3=new SparseDoubleMatrix2D(auxMat1.rows(),auxMat1.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat4=new SparseDoubleMatrix2D(auxMat2.rows(),auxMat2.rows(),0,0.7,0.75);
				
		int dim1=xty.rows();
		int dim2=xty.columns();
		
		System.out.println("+++Initialized auxiliary matrices+++");
				
		(svdTC.computeSparseInverse(xtx)).zMult(xty, auxMat1);
		
		System.out.println("+++Computed 1 inverse+++");
		
		(svdTC.computeSparseInverse(yty)).zMult(ytx, auxMat2);
		
		System.out.println("+++Computed Inverses+++");
		auxMat1.zMult(auxMat2,auxMat3);
		
		System.out.println("+++Entering SVD computation+++");
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
				
		SparseDoubleMatrix2D phiLT_LTL_phiL=new SparseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		SparseDoubleMatrix2D phiLT_LTR_phiR=new SparseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		SparseDoubleMatrix2D phiRT_RTL_phiL=new SparseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		SparseDoubleMatrix2D phiRT_RTR_phiR=new SparseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		
		SparseDoubleMatrix2D  ltl_phiL=new SparseDoubleMatrix2D(ltl.rows(),_opt.hiddenStateSize);
		SparseDoubleMatrix2D  ltr_phiR=new SparseDoubleMatrix2D(ltr.rows(),_opt.hiddenStateSize);
		SparseDoubleMatrix2D  rtl_phiL=new SparseDoubleMatrix2D(rtl.rows(),_opt.hiddenStateSize);
		SparseDoubleMatrix2D  rtr_phiR=new SparseDoubleMatrix2D(rtr.rows(),_opt.hiddenStateSize);
		
		
		computeCCA2(ltr,rtl, rtr,ltl,svdTC,_cpcaR2);
		
		wtl.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), WTLphiL);
		wtr.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), WTRphiR);
		
		phiLT=phiL.transpose();
		phiRT=phiR.transpose();
		
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(ltw, LTWphiL);
		MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(rtw, RTWphiR);
		
		WTLphiLWTRphiR=_cpcaR2.concatenateLR(WTLphiL,WTRphiR);
		phiLTLTWphiRTRTW=_cpcaR2.concatenateLRT(LTWphiL,RTWphiR);
		
		ltl.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), ltl_phiL);
		ltr.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), ltr_phiR);
		rtl.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), rtl_phiL);
		rtr.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), rtr_phiR);
		
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(ltl_phiL, phiLT_LTL_phiL);
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(ltr_phiR, phiLT_LTR_phiR);
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(rtl_phiL, phiRT_RTL_phiL);
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(rtr_phiR, phiRT_RTR_phiR);
		
		LRTLRphiLphiR=_cpcaR2.concatenateLRT(_cpcaR2.concatenateLR(phiLT_LTL_phiL,phiLT_LTR_phiR),_cpcaR2.concatenateLR(phiRT_RTL_phiL,phiRT_RTR_phiR));
		
		computeCCA2(WTLphiLWTRphiR,phiLTLTWphiRTRTW, LRTLRphiLphiR,wtw,svdTC,_cpcaR2);
		
	}

	private void computeCCATwoStepLRvsW(DenseDoubleMatrix2D ltr,
			DenseDoubleMatrix2D rtl, DenseDoubleMatrix2D ltl,
			DenseDoubleMatrix2D rtr, SparseDoubleMatrix2D wtw,
			DenseDoubleMatrix2D wtl, DenseDoubleMatrix2D wtr, DenseDoubleMatrix2D ltw, DenseDoubleMatrix2D rtw, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2) {
		
		
		
		DenseDoubleMatrix2D WTLphiL=new DenseDoubleMatrix2D(wtl.rows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D WTRphiR=new DenseDoubleMatrix2D(wtr.rows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D LTWphiL=new DenseDoubleMatrix2D(_opt.hiddenStateSize,wtl.rows());
		DenseDoubleMatrix2D RTWphiR=new DenseDoubleMatrix2D(_opt.hiddenStateSize,wtr.rows());
		DenseDoubleMatrix2D WTLphiLWTRphiR=new DenseDoubleMatrix2D(wtl.rows(),2*_opt.hiddenStateSize);
		DenseDoubleMatrix2D phiLTLTWphiRTRTW=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize,wtl.rows());
		
		DenseDoubleMatrix2D LRTLRphiLphiR=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize,2*_opt.hiddenStateSize);
				
		DenseDoubleMatrix2D phiLT_LTL_phiL=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		DenseDoubleMatrix2D phiLT_LTR_phiR=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		DenseDoubleMatrix2D phiRT_RTL_phiL=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		DenseDoubleMatrix2D phiRT_RTR_phiR=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		
		DenseDoubleMatrix2D  ltl_phiL=new DenseDoubleMatrix2D(ltl.rows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D  ltr_phiR=new DenseDoubleMatrix2D(ltr.rows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D  rtl_phiL=new DenseDoubleMatrix2D(rtl.rows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D  rtr_phiR=new DenseDoubleMatrix2D(rtr.rows(),_opt.hiddenStateSize);
		
		
		computeCCA2(ltr,rtl, rtr,ltl,svdTC,_cpcaR2);
		
		wtl.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), WTLphiL);
		wtr.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), WTRphiR);
		
		phiLT=phiL.transpose();
		phiRT=phiR.transpose();
		
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(ltw, LTWphiL);
		MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(rtw, RTWphiR);
		
		WTLphiLWTRphiR=_cpcaR2.concatenateLR(WTLphiL,WTRphiR);
		phiLTLTWphiRTRTW=_cpcaR2.concatenateLRT(LTWphiL,RTWphiR);
		
		ltl.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), ltl_phiL);
		ltr.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), ltr_phiR);
		rtl.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), rtl_phiL);
		rtr.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), rtr_phiR);
		
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(ltl_phiL, phiLT_LTL_phiL);
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(ltr_phiR, phiLT_LTR_phiR);
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(rtl_phiL, phiRT_RTL_phiL);
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(rtr_phiR, phiRT_RTR_phiR);
		
		LRTLRphiLphiR=_cpcaR2.concatenateLRT(_cpcaR2.concatenateLR(phiLT_LTL_phiL,phiLT_LTR_phiR),_cpcaR2.concatenateLR(phiRT_RTL_phiL,phiRT_RTR_phiR));
		
		computeCCA2(WTLphiLWTRphiR,phiLTLTWphiRTRTW, LRTLRphiLphiR,wtw,svdTC,_cpcaR2);
		
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
			
	//	if(!_opt.typeofDecomp.equals("TwoStepLRvsW")){
				ccaEig.writeObject(phiL);
				ccaEig.flush();
				ccaEig.close();
			
				ccaContext.writeObject(phiR);
				ccaContext.flush();
				ccaContext.close();
	
	/*	
		}else{
			ccaEig.writeObject(phiR);
			ccaEig.flush();
			ccaEig.close();
		
			ccaContext.writeObject(phiL);
			ccaContext.flush();
			ccaContext.close();
		}
		*/
			
			System.out.println("=======Serialized the CCA Variant Run=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
	
}
}
