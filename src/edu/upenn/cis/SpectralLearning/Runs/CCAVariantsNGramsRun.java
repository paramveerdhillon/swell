/**
 * @author Paramveer Dhillon
 *
 * Please send all comments, bug reports to dhillon@cis.upenn.edu
 * 
 *
 */


package edu.upenn.cis.SpectralLearning.Runs;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;

import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import Jama.Matrix;
import edu.upenn.cis.SpectralLearning.IO.Options;
import edu.upenn.cis.SpectralLearning.MathUtils.MatrixFormatConversion;
import edu.upenn.cis.SpectralLearning.MathUtils.SVDTemplates;
import edu.upenn.cis.SpectralLearning.SpectralRepresentations.ContextPCANGramsRepresentation;
import edu.upenn.cis.SpectralLearning.SpectralRepresentations.ContextPCARepresentation;

public class CCAVariantsNGramsRun {

	private Options _opt;
	private ContextPCANGramsRepresentation _cpcaR; 
	private int dim2=0;
	static final long serialVersionUID = 42L;
	HashMap<Double, Integer> _allDocs;
	Matrix phiL, phiR, phiLT,phiRT;
	double[] s;
	
	public CCAVariantsNGramsRun(Options opt,
			ContextPCANGramsRepresentation contextPCARep) {
		_opt=opt;
		_cpcaR=contextPCARep;
		//dim2=_opt.numLabels*(_opt.vocabSize+1);
		System.out.println("+++Entering CCA NGrams Compute+++");
		computeCCAVariantNGrams(_cpcaR);
		writeStats();
	}
	
	private void writeStats() {
		BufferedWriter writer=null;
		double sSum=0;
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream("Output_Files/CCAVariantNGramsStats"),"UTF8"));
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


private void computeCCAVariantNGrams(ContextPCANGramsRepresentation _cpcaR2) {
		
		if(_opt.depbigram)
			_cpcaR2.computeLRContextMatrices();
		else
			_cpcaR2.computeLRContextMatricesSingleVocab();
		SparseDoubleMatrix2D xtx,xty,ytx,wtr,rtw,ltw,wtl,wtw;
		
		
		
		SVDTemplates svdTC;
		
			svdTC=new SVDTemplates(_opt,dim2);
		System.out.println("+++Generated CCA Matrices+++");
		
		
		
		
		if(_opt.typeofDecomp.equals("2viewWvsL")){
			xty=_cpcaR2.getContextMatrix();
			ytx=_cpcaR2.getContextMatrixT();
			xtx=_cpcaR2.getWTWMatrix();
			
			
			computeCCA2NGrams(xtx,xty,ytx,svdTC,_cpcaR2,xtx.columns(),xty.columns());
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsR")){
			xty=_cpcaR2.getContextMatrix();
			ytx=_cpcaR2.getContextMatrixT();
			xtx=_cpcaR2.getWTWMatrix();
			computeCCA2NGrams(xtx,xty,ytx,svdTC,_cpcaR2,xtx.columns(),xty.columns());
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsLR")){
			xty=_cpcaR2.getContextMatrix();
			ytx=_cpcaR2.getContextMatrixT();
			xtx=_cpcaR2.getWTWMatrix();
			computeCCA2NGrams(xtx,xty,ytx,svdTC,_cpcaR2,xtx.columns(),xty.columns());
		}
		
		
		if(_opt.typeofDecomp.equals("TwoStepLRvsW")){
			
			
			wtl=_cpcaR2.getWL3gramMatrix();
			wtr=_cpcaR2.getWR3gramMatrix();
			ltw=_cpcaR2.getWLT3gramMatrix();
			rtw=_cpcaR2.getWRT3gramMatrix();
		
			if(_opt.numGrams==5){
				wtl=_cpcaR2.getWL5gramMatrix();
				wtr=_cpcaR2.getWR5gramMatrix();
				ltw=_cpcaR2.getWLT5gramMatrix();
				rtw=_cpcaR2.getWRT5gramMatrix();
			}
			
			wtw=_cpcaR2.getWTWMatrix();
	
			computeCCATwoStepLRvsWNGrams(wtl,wtr,ltw,rtw,wtw,svdTC,_cpcaR2,wtl.rows(),wtl.columns());
			
		}
	}
	
	

	private void computeCCA2NGrams(SparseDoubleMatrix2D xtx,
			SparseDoubleMatrix2D xty, SparseDoubleMatrix2D ytx,
			SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2,int dim1,int dim2) {
		
		SparseDoubleMatrix2D yty=new SparseDoubleMatrix2D(ytx.rows(),xty.columns());
		SparseDoubleMatrix2D auxMat1=new SparseDoubleMatrix2D(ytx.rows(),xtx.columns());
		SparseDoubleMatrix2D auxMat5=new SparseDoubleMatrix2D(xtx.columns(),ytx.rows());
		SparseDoubleMatrix2D auxMat2=new SparseDoubleMatrix2D(yty.rows(),ytx.columns());
		SparseDoubleMatrix2D auxMat3=new SparseDoubleMatrix2D(auxMat5.rows(),auxMat5.rows());
		SparseDoubleMatrix2D auxMat4=new SparseDoubleMatrix2D(auxMat2.rows(),auxMat2.rows());
				
				
		//ytx.zMult(svdTC.computeSparseInverse(xtx), auxMat1);
		//auxMat1.zMult(xty, yty);
		
		ytx.zMult(xty, yty);
		//auxMat1.zMult(xty, yty);
				
		
		
		(svdTC.computeSparseInverse(xtx)).zMult(xty, auxMat5);
		(svdTC.computeSparseInverse(yty)).zMult(ytx, auxMat2);
		auxMat5.zMult(auxMat2,auxMat3);
		phiL=svdTC.computeSVD_Tropp(auxMat3, _cpcaR2.getOmegaMatrix(auxMat3.columns()),dim1);
		s=svdTC.getSingularVals();
		auxMat2.zMult(auxMat5,auxMat4);
		phiR=svdTC.computeSVD_Tropp(auxMat4, _cpcaR2.getOmegaMatrix(auxMat4.columns()),dim2);
		
	}
	
	private void computeCCA2NGramsLR(SparseDoubleMatrix2D wtl,
			SparseDoubleMatrix2D wtr, SparseDoubleMatrix2D ltw,
			 SparseDoubleMatrix2D rtw,SparseDoubleMatrix2D wtw,SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2) {
		
		SparseDoubleMatrix2D ltl=new SparseDoubleMatrix2D(wtl.columns(),wtl.columns());
		SparseDoubleMatrix2D rtr=new SparseDoubleMatrix2D(wtr.columns(),wtr.columns());
		SparseDoubleMatrix2D ltr=new SparseDoubleMatrix2D(ltw.rows(),rtr.columns());
		SparseDoubleMatrix2D rtl=new SparseDoubleMatrix2D(rtw.rows(),ltl.columns());
		
		SparseDoubleMatrix2D auxMat1=new SparseDoubleMatrix2D(ltw.rows(),wtw.columns());
		SparseDoubleMatrix2D auxMat5=new SparseDoubleMatrix2D(ltl.rows(),ltr.columns());
		SparseDoubleMatrix2D auxMat6=new SparseDoubleMatrix2D(rtw.rows(),wtw.rows());
		SparseDoubleMatrix2D auxMat2=new SparseDoubleMatrix2D(rtr.rows(),rtl.columns());
		SparseDoubleMatrix2D auxMat3=new SparseDoubleMatrix2D(auxMat5.rows(),auxMat5.rows());
		SparseDoubleMatrix2D auxMat4=new SparseDoubleMatrix2D(auxMat2.rows(),auxMat2.rows());
				
				
		//ltw.zMult(svdTC.computeSparseInverse(wtw), auxMat1);
		//auxMat1.zMult(wtl, ltl);
		
		//rtw.zMult(svdTC.computeSparseInverse(wtw), auxMat6);
		//auxMat6.zMult(wtr, rtr);
		
		ltw.zMult(wtl, ltl);
		
		rtw.zMult(wtr, rtr);
		
		
		
		ltw.zMult(wtr, ltr);
		
		rtw.zMult(wtl, rtl);
		
		
		(svdTC.computeSparseInverse(ltl)).zMult(ltr, auxMat5);
		(svdTC.computeSparseInverse(rtr)).zMult(rtl, auxMat2);
		auxMat5.zMult(auxMat2,auxMat3);
		phiL=svdTC.computeSVD_Tropp(auxMat3, _cpcaR2.getOmegaMatrix(auxMat3.columns()),auxMat3.columns());
		
		auxMat2.zMult(auxMat5,auxMat4);
		phiR=svdTC.computeSVD_Tropp(auxMat4, _cpcaR2.getOmegaMatrix(auxMat4.columns()),auxMat4.columns());
		
	}
	
	private void computeCCATwoStepLRvsWNGrams(SparseDoubleMatrix2D wtl,
			SparseDoubleMatrix2D wtr, SparseDoubleMatrix2D ltw,
			SparseDoubleMatrix2D rtw,SparseDoubleMatrix2D wtw,
			SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2,int dim1,int dim2) {
		
		
		SparseDoubleMatrix2D wtlphiL=new SparseDoubleMatrix2D(wtl.rows(),_opt.hiddenStateSize);
		SparseDoubleMatrix2D wtrphiR=new SparseDoubleMatrix2D(wtr.rows(),_opt.hiddenStateSize);
		SparseDoubleMatrix2D ltwphiLT=new SparseDoubleMatrix2D(_opt.hiddenStateSize,wtl.rows());
		SparseDoubleMatrix2D rtwphiRT=new SparseDoubleMatrix2D(_opt.hiddenStateSize,wtr.rows());
		SparseDoubleMatrix2D wtLphiLRphiR=new SparseDoubleMatrix2D(wtl.rows(),wtlphiL.columns()+wtrphiR.columns());
		SparseDoubleMatrix2D wtLphiLRphiRT=new SparseDoubleMatrix2D(wtlphiL.columns()+wtrphiR.columns(),wtl.rows());
				
		
		
		
		computeCCA2NGramsLR(wtl,wtr,ltw,rtw,wtw,svdTC,_cpcaR2);
		
		wtl.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), wtlphiL);
		wtr.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), wtrphiR);
		
		phiLT=phiL.transpose();
		phiRT=phiR.transpose();
		
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(ltw, ltwphiLT);
		MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(rtw, rtwphiRT);
		
		wtLphiLRphiR=_cpcaR2.concatenateLR(wtlphiL,wtrphiR);
		wtLphiLRphiRT=_cpcaR2.concatenateLRT(ltwphiLT,rtwphiRT);
		
		computeCCA2NGrams( wtw,wtLphiLRphiR,wtLphiLRphiRT,svdTC,_cpcaR2,wtLphiLRphiR.rows(),wtLphiLRphiRT.rows());
		
	}


/*
	private void computeCCA3NGrams(SparseDoubleMatrix2D view1,
			SparseDoubleMatrix2D view1t, SparseDoubleMatrix2D view2,
			SparseDoubleMatrix2D view2t, SparseDoubleMatrix2D view3,
			SparseDoubleMatrix2D view3t, SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2) {
		// TODO Auto-generated method stub
		
	}
*/

	

	public Matrix getPhiL(){
		return phiL;
	}
	
	public Matrix getPhiR(){
		return phiR;
	}

	
	
public void serializeCCAVariantsNGramsRun() {
		
		
		String contextDict=_opt.serializeRun+"Context";
		File fContext= new File(contextDict);
		
		String eigDict=_opt.serializeRun+"Eig";
		File fEig= new File(eigDict);
			
		try{
			ObjectOutput ccaEig=new ObjectOutputStream(new FileOutputStream(fEig));
			ObjectOutput ccaContext=new ObjectOutputStream(new FileOutputStream(fContext));
			
		//if(!_opt.typeofDecomp.equals("TwoStepLRvsW")){
				ccaEig.writeObject(phiL);
				ccaEig.flush();
				ccaEig.close();
			
				ccaContext.writeObject(phiR);
				ccaContext.flush();
				ccaContext.close();
	/*	}else{
			ccaEig.writeObject(phiR);
			ccaEig.flush();
			ccaEig.close();
		
			ccaContext.writeObject(phiL);
			ccaContext.flush();
			ccaContext.close();
		} */
			
			System.out.println("=======Serialized the CCA Variant NGrams Run=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
	
}
}

