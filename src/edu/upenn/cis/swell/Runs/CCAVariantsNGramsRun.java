/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */


package edu.upenn.cis.swell.Runs;

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
import edu.upenn.cis.swell.IO.Options;
import edu.upenn.cis.swell.MathUtils.MatrixFormatConversion;
import edu.upenn.cis.swell.MathUtils.SVDTemplates;
import edu.upenn.cis.swell.SpectralRepresentations.ContextPCANGramsRepresentation;

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
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.outputDir+"CCAVariantNGramsStats"),"UTF8"));
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
		SparseDoubleMatrix2D xtx,xty,ytx,wtr,rtw,ltw,wtl,wtw,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T
		,L1L3_OR_R1R3_L1R2,L1L3_OR_R1R3_L1R2T,L1L4_OR_R1R4_L2R1,L1L4_OR_R1R4_L2R1T,L2L3_OR_R2R3_L2R2,L2L3_OR_R2R3_L2R2T,
		L2L4_OR_R2R4_L1L2,L2L4_OR_R2R4_L1L2T,L3L4_OR_R3R4_R1R2,L3L4_OR_R3R4_R1R2T;
		
		
		
		SVDTemplates svdTC;
		
			svdTC=new SVDTemplates(_opt,dim2);
		System.out.println("+++Generated CCA Matrices+++");
		
		
		
		
		if(_opt.typeofDecomp.equals("2viewWvsL")){
			xty=_cpcaR2.getContextMatrix();
			ytx=_cpcaR2.getContextMatrixT();
			xtx=_cpcaR2.getWTWMatrix();
			L1L2_OR_R1R2_L1R1=_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesv();
			L1L2_OR_R1R2_L1R1T=_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesvT();
			
			if(_opt.numGrams==3){
			computeCCA2NGrams(xtx,xty,ytx,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T,svdTC,_cpcaR2,xtx.columns(),xty.columns());
			}
			
			if(_opt.numGrams==5){
				L1L3_OR_R1R3_L1R2= _cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesv();
				L1L3_OR_R1R3_L1R2T =_cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesvT();
				
				L1L4_OR_R1R4_L2R1= _cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesv();
				L1L4_OR_R1R4_L2R1T= _cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesvT();
				
				L2L3_OR_R2R3_L2R2= _cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesv();
				L2L3_OR_R2R3_L2R2T= _cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesvT();
				
				L2L4_OR_R2R4_L1L2= _cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesv();
				L2L4_OR_R2R4_L1L2T= _cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesvT();
				
				L3L4_OR_R3R4_R1R2= _cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesv();
				L3L4_OR_R3R4_R1R2T= _cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesvT();
				
				computeCCA2NGrams(xtx,xty,ytx,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T
						,L1L3_OR_R1R3_L1R2,L1L3_OR_R1R3_L1R2T,L1L4_OR_R1R4_L2R1,L1L4_OR_R1R4_L2R1T,L2L3_OR_R2R3_L2R2,L2L3_OR_R2R3_L2R2T,
						L2L4_OR_R2R4_L1L2,L2L4_OR_R2R4_L1L2T,L3L4_OR_R3R4_R1R2,L3L4_OR_R3R4_R1R2T,svdTC,_cpcaR2,xtx.columns(),xty.columns());
				
			}
			
			
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsR")){
			xty=_cpcaR2.getContextMatrix();
			ytx=_cpcaR2.getContextMatrixT();
			xtx=_cpcaR2.getWTWMatrix();
			L1L2_OR_R1R2_L1R1=_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesv();
			L1L2_OR_R1R2_L1R1T=_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesvT();
			
			if(_opt.numGrams==3 || _opt.numGrams==2){
				computeCCA2NGrams(xtx,xty,ytx,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T,svdTC,_cpcaR2,xtx.columns(),xty.columns());
				}
			
			if(_opt.numGrams==5){
				L1L3_OR_R1R3_L1R2= _cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesv();
				L1L3_OR_R1R3_L1R2T =_cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesvT();
				
				L1L4_OR_R1R4_L2R1= _cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesv();
				L1L4_OR_R1R4_L2R1T= _cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesvT();
				
				L2L3_OR_R2R3_L2R2= _cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesv();
				L2L3_OR_R2R3_L2R2T= _cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesvT();
				
				L2L4_OR_R2R4_L1L2= _cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesv();
				L2L4_OR_R2R4_L1L2T= _cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesvT();
				
				L3L4_OR_R3R4_R1R2= _cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesv();
				L3L4_OR_R3R4_R1R2T= _cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesvT();
				
				computeCCA2NGrams(xtx,xty,ytx,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T
						,L1L3_OR_R1R3_L1R2,L1L3_OR_R1R3_L1R2T,L1L4_OR_R1R4_L2R1,L1L4_OR_R1R4_L2R1T,L2L3_OR_R2R3_L2R2,L2L3_OR_R2R3_L2R2T,
						L2L4_OR_R2R4_L1L2,L2L4_OR_R2R4_L1L2T,L3L4_OR_R3R4_R1R2,L3L4_OR_R3R4_R1R2T,svdTC,_cpcaR2,xtx.columns(),xty.columns());
				
			}
			
			
			
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsLR")){
			xty=_cpcaR2.getContextMatrix();
			ytx=_cpcaR2.getContextMatrixT();
			xtx=_cpcaR2.getWTWMatrix();
			L1L2_OR_R1R2_L1R1=_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesv();
			L1L2_OR_R1R2_L1R1T=_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesvT();
			
			if(_opt.numGrams==3){
				computeCCA2NGrams(xtx,xty,ytx,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T,svdTC,_cpcaR2,xtx.columns(),xty.columns());
				}
			
			if(_opt.numGrams==5){
				L1L3_OR_R1R3_L1R2= _cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesv();
				L1L3_OR_R1R3_L1R2T =_cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesvT();
				
				L1L4_OR_R1R4_L2R1= _cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesv();
				L1L4_OR_R1R4_L2R1T= _cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesvT();
				
				L2L3_OR_R2R3_L2R2= _cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesv();
				L2L3_OR_R2R3_L2R2T= _cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesvT();
				
				L2L4_OR_R2R4_L1L2= _cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesv();
				L2L4_OR_R2R4_L1L2T= _cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesvT();
				
				L3L4_OR_R3R4_R1R2= _cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesv();
				L3L4_OR_R3R4_R1R2T= _cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesvT();
				
				computeCCA2NGrams(xtx,xty,ytx,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T
						,L1L3_OR_R1R3_L1R2,L1L3_OR_R1R3_L1R2T,L1L4_OR_R1R4_L2R1,L1L4_OR_R1R4_L2R1T,L2L3_OR_R2R3_L2R2,L2L3_OR_R2R3_L2R2T,
						L2L4_OR_R2R4_L1L2,L2L4_OR_R2R4_L1L2T,L3L4_OR_R3R4_R1R2,L3L4_OR_R3R4_R1R2T,svdTC,_cpcaR2,xtx.columns(),xty.columns());
				
			}
			
		}
		
		
		if(_opt.typeofDecomp.equals("TwoStepLRvsW")){
			
			
			wtl=_cpcaR2.getWL3gramMatrix();
			wtr=_cpcaR2.getWR3gramMatrix();
			ltw=_cpcaR2.getWLT3gramMatrix();
			rtw=_cpcaR2.getWRT3gramMatrix();
			L1L2_OR_R1R2_L1R1=_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesv();
			L1L2_OR_R1R2_L1R1T=_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesvT();
			wtw=_cpcaR2.getWTWMatrix();
			
			
			if(_opt.numGrams==3){
				computeCCATwoStepLRvsWNGrams(wtl,wtr,ltw,rtw,wtw,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T,null,null,null,null,null,null,null,null,null,null,svdTC,_cpcaR2,wtl.rows(),wtl.columns());
			}
			
			if(_opt.numGrams==5){
				wtl=_cpcaR2.getWL5gramMatrix();
				wtr=_cpcaR2.getWR5gramMatrix();
				ltw=_cpcaR2.getWLT5gramMatrix();
				rtw=_cpcaR2.getWRT5gramMatrix();
				
				
				L1L3_OR_R1R3_L1R2= _cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesv();
				L1L3_OR_R1R3_L1R2T =_cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesvT();
				
				L1L4_OR_R1R4_L2R1= _cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesv();
				L1L4_OR_R1R4_L2R1T= _cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesvT();
				
				L2L3_OR_R2R3_L2R2= _cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesv();
				L2L3_OR_R2R3_L2R2T= _cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesvT();
				
				L2L4_OR_R2R4_L1L2= _cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesv();
				L2L4_OR_R2R4_L1L2T= _cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesvT();
				
				L3L4_OR_R3R4_R1R2= _cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesv();
				L3L4_OR_R3R4_R1R2T= _cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesvT();
				
				computeCCATwoStepLRvsWNGrams(wtl,wtr,ltw,rtw,wtw,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T
						,L1L3_OR_R1R3_L1R2,L1L3_OR_R1R3_L1R2T,L1L4_OR_R1R4_L2R1,L1L4_OR_R1R4_L2R1T,L2L3_OR_R2R3_L2R2,L2L3_OR_R2R3_L2R2T,
						L2L4_OR_R2R4_L1L2,L2L4_OR_R2R4_L1L2T,L3L4_OR_R3R4_R1R2,L3L4_OR_R3R4_R1R2T,svdTC,_cpcaR2,wtl.rows(),wtl.columns());
				
				
			}
			
		}
	}
	
	

	private void computeCCA2NGrams(SparseDoubleMatrix2D xtx,
			SparseDoubleMatrix2D xty, SparseDoubleMatrix2D ytx,SparseDoubleMatrix2D L1L2_OR_R1R2_L1R1,SparseDoubleMatrix2D L1L2_OR_R1R2_L1R1T,
			SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2,int dim1,int dim2) {
		
		SparseDoubleMatrix2D yty=new SparseDoubleMatrix2D(ytx.rows(),xty.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat5=new SparseDoubleMatrix2D(xtx.columns(),ytx.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat2=new SparseDoubleMatrix2D(yty.rows(),ytx.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat3=new SparseDoubleMatrix2D(auxMat5.rows(),auxMat5.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat4=new SparseDoubleMatrix2D(auxMat2.rows(),auxMat2.rows(),0,0.7,0.75);
				
		
		if(_opt.numGrams==3){
			
			
			SparseDoubleMatrix2D topM=new SparseDoubleMatrix2D(xtx.rows(),xtx.columns()+L1L2_OR_R1R2_L1R1.columns());
			SparseDoubleMatrix2D bottomM=new SparseDoubleMatrix2D(topM.rows(),topM.columns());
			
			
			topM=_cpcaR2.concatenateLR(xtx,L1L2_OR_R1R2_L1R1);
			bottomM=_cpcaR2.concatenateLR(L1L2_OR_R1R2_L1R1T,xtx);
			yty=_cpcaR2.concatenateLRT(topM,bottomM);
			if(_opt.typeofDecomp.equals("TwoStepLRvsW"))
					{
						SparseDoubleMatrix2D topM1=new SparseDoubleMatrix2D(phiLT.getRowDimension(),phiLT.getRowDimension(),0,0.7,0.75);
						SparseDoubleMatrix2D bottomM1=new SparseDoubleMatrix2D(phiLT.getRowDimension(),phiLT.getRowDimension(),0,0.7,0.75);
				
						SparseDoubleMatrix2D phiLTxtx=new SparseDoubleMatrix2D(phiLT.getRowDimension(),xtx.columns(),0,0.7,0.75);
						SparseDoubleMatrix2D phiLTxtxphiL=new SparseDoubleMatrix2D(phiLT.getRowDimension(),phiLT.getRowDimension(),0,0.7,0.75);
						SparseDoubleMatrix2D phiLTLR=new SparseDoubleMatrix2D(phiLT.getRowDimension(),xtx.columns(),0,0.7,0.75);
						SparseDoubleMatrix2D phiLTLRphiR=new SparseDoubleMatrix2D(phiLT.getRowDimension(),phiLT.getRowDimension(),0,0.7,0.75);
						SparseDoubleMatrix2D phiRTLR=new SparseDoubleMatrix2D(phiLT.getRowDimension(),xtx.columns(),0,0.7,0.75);
						SparseDoubleMatrix2D phiRTLRphiL=new SparseDoubleMatrix2D(phiLT.getRowDimension(),phiLT.getRowDimension(),0,0.7,0.75);
				
						phiLT=phiL.transpose();
						phiRT=phiR.transpose();
						MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(xtx, phiLTxtx);
						phiLTxtx.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), phiLTxtxphiL);
						
						MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(L1L2_OR_R1R2_L1R1, phiLTLR);
						phiLTLR.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), phiLTLRphiR);
						
						MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(L1L2_OR_R1R2_L1R1T, phiRTLR);
						phiRTLR.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), phiRTLRphiL);
				
				
						topM1=_cpcaR2.concatenateLR(phiLTxtxphiL,phiLTLRphiR);
						bottomM1=_cpcaR2.concatenateLR(phiRTLRphiL,phiLTxtxphiL);
						
						yty=_cpcaR2.concatenateLRT(topM1,bottomM1);
					}
					
			
		}
		else{
			yty=xtx;
		}
		
		
		(svdTC.computeSparseInverse(xtx)).zMult(xty, auxMat5);
		(svdTC.computeSparseInverse(yty)).zMult(ytx, auxMat2);
		auxMat5.zMult(auxMat2,auxMat3);
		phiL=svdTC.computeSVD_Tropp(auxMat3, _cpcaR2.getOmegaMatrix(auxMat3.columns()),dim1);
		s=svdTC.getSingularVals();
		if(!_opt.typeofDecomp.equals("TwoStepLRvsW")){
			auxMat2.zMult(auxMat5,auxMat4);
			phiR=svdTC.computeSVD_Tropp(auxMat4, _cpcaR2.getOmegaMatrix(auxMat4.columns()),dim2);
		}
		
	}
	
	
	private void computeCCA2NGrams(SparseDoubleMatrix2D xtx,
			SparseDoubleMatrix2D xty, SparseDoubleMatrix2D ytx,SparseDoubleMatrix2D L1L2_OR_R1R2_L1R1,
			SparseDoubleMatrix2D L1L2_OR_R1R2_L1R1T, SparseDoubleMatrix2D L1L3_OR_R1R3_L1R2,SparseDoubleMatrix2D L1L3_OR_R1R3_L1R2T,
			SparseDoubleMatrix2D L1L4_OR_R1R4_L2R1, SparseDoubleMatrix2D L1L4_OR_R1R4_L2R1T,SparseDoubleMatrix2D L2L3_OR_R2R3_L2R2,
			SparseDoubleMatrix2D L2L3_OR_R2R3_L2R2T, SparseDoubleMatrix2D L2L4_OR_R2R4_L1L2,SparseDoubleMatrix2D L2L4_OR_R2R4_L1L2T,
			SparseDoubleMatrix2D L3L4_OR_R3R4_R1R2, SparseDoubleMatrix2D L3L4_OR_R3R4_R1R2T,
			SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2,int dim1,int dim2) {
			
		SparseDoubleMatrix2D yty=new SparseDoubleMatrix2D(ytx.rows(),xty.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat5=new SparseDoubleMatrix2D(xtx.columns(),ytx.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat2=new SparseDoubleMatrix2D(yty.rows(),ytx.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat3=new SparseDoubleMatrix2D(auxMat5.rows(),auxMat5.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat4=new SparseDoubleMatrix2D(auxMat2.rows(),auxMat2.rows(),0,0.7,0.75);
				
		
		SparseDoubleMatrix2D row1=new SparseDoubleMatrix2D(xtx.rows(),xtx.columns()+L1L2_OR_R1R2_L1R1.columns()+ L1L3_OR_R1R3_L1R2.columns()+ L1L4_OR_R1R4_L2R1.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D row2=new SparseDoubleMatrix2D(xtx.rows(),xtx.columns()+L1L2_OR_R1R2_L1R1.columns()+ L1L3_OR_R1R3_L1R2.columns()+ L1L4_OR_R1R4_L2R1.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D row3=new SparseDoubleMatrix2D(xtx.rows(),xtx.columns()+L1L2_OR_R1R2_L1R1.columns()+ L1L3_OR_R1R3_L1R2.columns()+ L1L4_OR_R1R4_L2R1.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D row4=new SparseDoubleMatrix2D(xtx.rows(),xtx.columns()+L1L2_OR_R1R2_L1R1.columns()+ L1L3_OR_R1R3_L1R2.columns()+ L1L4_OR_R1R4_L2R1.columns(),0,0.7,0.75);
		
		
		row1=_cpcaR2.concatenateMultiRow(xtx,L1L2_OR_R1R2_L1R1, L1L3_OR_R1R3_L1R2, L1L4_OR_R1R4_L2R1);
		row2=_cpcaR2.concatenateMultiRow(L1L2_OR_R1R2_L1R1T,xtx, L2L3_OR_R2R3_L2R2, L2L4_OR_R2R4_L1L2);
		row3=_cpcaR2.concatenateMultiRow(L1L3_OR_R1R3_L1R2T,L2L3_OR_R2R3_L2R2T ,xtx,L3L4_OR_R3R4_R1R2);
		row4=_cpcaR2.concatenateMultiRow(L1L4_OR_R1R4_L2R1T,L2L4_OR_R2R4_L1L2T,L3L4_OR_R3R4_R1R2T, xtx);
		yty=_cpcaR2.concatenateMultiCol(row1,row2,row3,row4);
		
		
		if(_opt.typeofDecomp.equals("TwoStepLRvsW"))
		{
			
			SparseDoubleMatrix2D row11=new SparseDoubleMatrix2D(phiLT.getRowDimension(),phiLT.getRowDimension(),0,0.7,0.75);
			SparseDoubleMatrix2D row21=new SparseDoubleMatrix2D(phiLT.getRowDimension(),phiLT.getRowDimension(),0,0.7,0.75);
			
			
			SparseDoubleMatrix2D aux=new SparseDoubleMatrix2D(phiLT.getRowDimension(),xtx.columns()+L2L4_OR_R2R4_L1L2.columns(),0,0.7,0.75);
			
			SparseDoubleMatrix2D aux1=new SparseDoubleMatrix2D(xtx.rows(),xtx.columns()+L2L4_OR_R2R4_L1L2.columns(),0,0.7,0.75);
			SparseDoubleMatrix2D aux2=new SparseDoubleMatrix2D(xtx.rows(),xtx.columns()+L2L4_OR_R2R4_L1L2.columns(),0,0.7,0.75);
			
			SparseDoubleMatrix2D blockRow12Col12=new SparseDoubleMatrix2D(xtx.rows()+L2L4_OR_R2R4_L1L2.rows(),xtx.columns()+L2L4_OR_R2R4_L1L2.columns(),0,0.7,0.75);
			SparseDoubleMatrix2D blockRow12Col34=new SparseDoubleMatrix2D(xtx.rows()+L2L4_OR_R2R4_L1L2.rows(),xtx.columns()+L2L4_OR_R2R4_L1L2.columns(),0,0.7,0.75);
			SparseDoubleMatrix2D blockRow34Col12=new SparseDoubleMatrix2D(xtx.rows()+L2L4_OR_R2R4_L1L2.rows(),xtx.columns()+L2L4_OR_R2R4_L1L2.columns(),0,0.7,0.75);
			SparseDoubleMatrix2D blockRow34Col34=new SparseDoubleMatrix2D(xtx.rows()+L2L4_OR_R2R4_L1L2.rows(),xtx.columns()+L2L4_OR_R2R4_L1L2.columns(),0,0.7,0.75);
			SparseDoubleMatrix2D blockRow12Col12Phi=new SparseDoubleMatrix2D(phiLT.getRowDimension(),phiLT.getRowDimension(),0,0.7,0.75);
			SparseDoubleMatrix2D blockRow12Col34Phi=new SparseDoubleMatrix2D(phiLT.getRowDimension(),phiLT.getRowDimension(),0,0.7,0.75);
			SparseDoubleMatrix2D blockRow34Col12Phi=new SparseDoubleMatrix2D(phiLT.getRowDimension(),phiLT.getRowDimension(),0,0.7,0.75);
			SparseDoubleMatrix2D  blockRow34Col34Phi=new SparseDoubleMatrix2D(phiLT.getRowDimension(),phiLT.getRowDimension(),0,0.7,0.75);
		
			
			aux1=_cpcaR2.concatenateLR(xtx,L2L4_OR_R2R4_L1L2);
			aux2=_cpcaR2.concatenateLR(L2L4_OR_R2R4_L1L2T,xtx);
			blockRow12Col12=_cpcaR2.concatenateLRT(aux1,aux2);
			
			
			aux1=_cpcaR2.concatenateLR(xtx,L3L4_OR_R3R4_R1R2);
			aux2=_cpcaR2.concatenateLR(L3L4_OR_R3R4_R1R2T,xtx);
			blockRow34Col34=_cpcaR2.concatenateLRT(aux1,aux2);
			
			aux1=_cpcaR2.concatenateLR(L1L2_OR_R1R2_L1R1,L1L3_OR_R1R3_L1R2);
			aux2=_cpcaR2.concatenateLR(L1L3_OR_R1R3_L1R2T,L2L3_OR_R2R3_L2R2);
			blockRow12Col34=_cpcaR2.concatenateLRT(aux1,aux2);
			
			
			aux1=_cpcaR2.concatenateLR(L1L2_OR_R1R2_L1R1T,L1L4_OR_R1R4_L2R1T);
			aux2=_cpcaR2.concatenateLR(L1L3_OR_R1R3_L1R2T,L2L3_OR_R2R3_L2R2T);
			blockRow34Col12=_cpcaR2.concatenateLRT(aux1,aux2);
			
			
			
			
			phiLT=phiL.transpose();
			phiRT=phiR.transpose();
			MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(blockRow12Col12, aux);
			aux.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), blockRow12Col12Phi);
			
			MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(blockRow34Col34, aux);
			aux.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), blockRow34Col34Phi);
			
			MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(blockRow34Col12, aux);
			aux.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), blockRow34Col12Phi);
			
			MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(blockRow12Col34, aux);
			aux.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), blockRow12Col34Phi);
			
			
	
			row11=_cpcaR2.concatenateLR(blockRow12Col12Phi,blockRow12Col34Phi);
			row21=_cpcaR2.concatenateLR(blockRow34Col12Phi,blockRow34Col34Phi);
			
			yty=_cpcaR2.concatenateLRT(row11,row21);
			

		}
		

		
		(svdTC.computeSparseInverse(xtx)).zMult(xty, auxMat5);
		(svdTC.computeSparseInverse(yty)).zMult(ytx, auxMat2);
		auxMat5.zMult(auxMat2,auxMat3);
		phiL=svdTC.computeSVD_Tropp(auxMat3, _cpcaR2.getOmegaMatrix(auxMat3.columns()),dim1);
		s=svdTC.getSingularVals();
		
		if(!_opt.typeofDecomp.equals("TwoStepLRvsW")){
			auxMat2.zMult(auxMat5,auxMat4);
			phiR=svdTC.computeSVD_Tropp(auxMat4, _cpcaR2.getOmegaMatrix(auxMat4.columns()),dim2);
		}
		
	}
	
	private void computeCCA2NGramsLR(SparseDoubleMatrix2D ltr,
			SparseDoubleMatrix2D rtl, SparseDoubleMatrix2D wtw,
			 SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2) {
		
		
		SparseDoubleMatrix2D auxMat5=new SparseDoubleMatrix2D(wtw.rows(),ltr.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat2=new SparseDoubleMatrix2D(wtw.rows(),rtl.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat3=new SparseDoubleMatrix2D(auxMat5.rows(),auxMat5.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat4=new SparseDoubleMatrix2D(auxMat2.rows(),auxMat2.rows(),0,0.7,0.75);
				
				
		(svdTC.computeSparseInverse(wtw)).zMult(ltr, auxMat5);
		(svdTC.computeSparseInverse(wtw)).zMult(rtl, auxMat2);
		auxMat5.zMult(auxMat2,auxMat3);
		phiL=svdTC.computeSVD_Tropp(auxMat3, _cpcaR2.getOmegaMatrix(auxMat3.columns()),auxMat3.columns());
		
		auxMat2.zMult(auxMat5,auxMat4);
		phiR=svdTC.computeSVD_Tropp(auxMat4, _cpcaR2.getOmegaMatrix(auxMat4.columns()),auxMat4.columns());
		
	}
	
	private void computeCCATwoStepLRvsWNGrams(SparseDoubleMatrix2D wtl,
			SparseDoubleMatrix2D wtr, SparseDoubleMatrix2D ltw,
			SparseDoubleMatrix2D rtw,SparseDoubleMatrix2D wtw,
			SparseDoubleMatrix2D l1l2_OR_R1R2_L1R1, SparseDoubleMatrix2D l1l2_OR_R1R2_L1R1T, SparseDoubleMatrix2D l1l3_OR_R1R3_L1R2, SparseDoubleMatrix2D l1l3_OR_R1R3_L1R2T, SparseDoubleMatrix2D l1l4_OR_R1R4_L2R1, SparseDoubleMatrix2D l1l4_OR_R1R4_L2R1T, SparseDoubleMatrix2D l2l3_OR_R2R3_L2R2, SparseDoubleMatrix2D l2l3_OR_R2R3_L2R2T, SparseDoubleMatrix2D l2l4_OR_R2R4_L1L2, SparseDoubleMatrix2D l2l4_OR_R2R4_L1L2T, SparseDoubleMatrix2D l3l4_OR_R3R4_R1R2, SparseDoubleMatrix2D l3l4_OR_R3R4_R1R2T, SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2,int dim1,int dim2) {
		
		
		SparseDoubleMatrix2D wtlphiL=new SparseDoubleMatrix2D(wtl.rows(),_opt.hiddenStateSize,0,0.7,0.75);
		SparseDoubleMatrix2D wtrphiR=new SparseDoubleMatrix2D(wtr.rows(),_opt.hiddenStateSize,0,0.7,0.75);
		SparseDoubleMatrix2D ltwphiLT=new SparseDoubleMatrix2D(_opt.hiddenStateSize,wtl.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D rtwphiRT=new SparseDoubleMatrix2D(_opt.hiddenStateSize,wtr.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D wtLphiLRphiR=new SparseDoubleMatrix2D(wtl.rows(),wtlphiL.columns()+wtrphiR.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D wtLphiLRphiRT=new SparseDoubleMatrix2D(wtlphiL.columns()+wtrphiR.columns(),wtl.rows(),0,0.7,0.75);
				
		
		if(_opt.numGrams==3){
			computeCCA2NGramsLR(l1l2_OR_R1R2_L1R1,l1l2_OR_R1R2_L1R1T,wtw,svdTC,_cpcaR2);
		}
		
		
		
		if(_opt.numGrams==5){
			computeCCA2NGramsLR(l1l2_OR_R1R2_L1R1,l1l2_OR_R1R2_L1R1T,l1l3_OR_R1R3_L1R2,l1l3_OR_R1R3_L1R2T,l1l4_OR_R1R4_L2R1,l1l4_OR_R1R4_L2R1T,l2l3_OR_R2R3_L2R2,l2l3_OR_R2R3_L2R2T,
					l2l4_OR_R2R4_L1L2,l2l4_OR_R2R4_L1L2T,l3l4_OR_R3R4_R1R2,l3l4_OR_R3R4_R1R2T,wtw,svdTC,_cpcaR2);
		}
		
		wtl.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), wtlphiL);
		wtr.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), wtrphiR);
		
		phiLT=phiL.transpose();
		phiRT=phiR.transpose();
		
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(ltw, ltwphiLT);
		MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(rtw, rtwphiRT);
		
		wtLphiLRphiR=_cpcaR2.concatenateLR(wtlphiL,wtrphiR);
		wtLphiLRphiRT=_cpcaR2.concatenateLRT(ltwphiLT,rtwphiRT);
		
		if(_opt.numGrams==3){
			computeCCA2NGrams( wtw,wtLphiLRphiR,wtLphiLRphiRT,l1l2_OR_R1R2_L1R1, l1l2_OR_R1R2_L1R1T,svdTC,_cpcaR2,wtLphiLRphiR.rows(),wtLphiLRphiRT.rows());
		}
		if(_opt.numGrams==5){
			computeCCA2NGrams(wtw,wtLphiLRphiR,wtLphiLRphiRT,l1l2_OR_R1R2_L1R1,l1l2_OR_R1R2_L1R1T
					,l1l3_OR_R1R3_L1R2,l1l3_OR_R1R3_L1R2T,l1l4_OR_R1R4_L2R1,l1l4_OR_R1R4_L2R1T,l2l3_OR_R2R3_L2R2,l2l3_OR_R2R3_L2R2T,
					l2l4_OR_R2R4_L1L2,l2l4_OR_R2R4_L1L2T,l3l4_OR_R3R4_R1R2,l3l4_OR_R3R4_R1R2T,svdTC,_cpcaR2,wtLphiLRphiR.rows(),wtLphiLRphiRT.rows());
			
		}
	}


/*
	private void computeCCA3NGrams(SparseDoubleMatrix2D view1,
			SparseDoubleMatrix2D view1t, SparseDoubleMatrix2D view2,
			SparseDoubleMatrix2D view2t, SparseDoubleMatrix2D view3,
			SparseDoubleMatrix2D view3t, SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2) {
		// TODO Auto-generated method stub
		
	}
*/

	

	private void computeCCA2NGramsLR(SparseDoubleMatrix2D l1l2_OR_R1R2_L1R1,
			SparseDoubleMatrix2D l1l2_OR_R1R2_L1R1T,
			SparseDoubleMatrix2D l1l3_OR_R1R3_L1R2,
			SparseDoubleMatrix2D l1l3_OR_R1R3_L1R2T,
			SparseDoubleMatrix2D l1l4_OR_R1R4_L2R1,
			SparseDoubleMatrix2D l1l4_OR_R1R4_L2R1T,
			SparseDoubleMatrix2D l2l3_OR_R2R3_L2R2,
			SparseDoubleMatrix2D l2l3_OR_R2R3_L2R2T,
			SparseDoubleMatrix2D l2l4_OR_R2R4_L1L2,
			SparseDoubleMatrix2D l2l4_OR_R2R4_L1L2T,
			SparseDoubleMatrix2D l3l4_OR_R3R4_R1R2,
			SparseDoubleMatrix2D l3l4_OR_R3R4_R1R2T, SparseDoubleMatrix2D wtw,
			SVDTemplates svdTC, ContextPCANGramsRepresentation _cpcaR2) {
		

		
		SparseDoubleMatrix2D rtr=new SparseDoubleMatrix2D(l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D ltl=new SparseDoubleMatrix2D(l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D ltr=new SparseDoubleMatrix2D(l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D rtl=new SparseDoubleMatrix2D(l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),0,0.7,0.75);
		
		
		
		SparseDoubleMatrix2D auxMat5=new SparseDoubleMatrix2D(l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat2=new SparseDoubleMatrix2D(l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat3=new SparseDoubleMatrix2D(l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D auxMat4=new SparseDoubleMatrix2D(l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),0,0.7,0.75);
				
		
		SparseDoubleMatrix2D row1=new SparseDoubleMatrix2D(l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),0,0.7,0.75);
		SparseDoubleMatrix2D row2=new SparseDoubleMatrix2D(l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(),0,0.7,0.75);
		
		row1=_cpcaR2.concatenateLR(wtw,l2l4_OR_R2R4_L1L2);
		row2=_cpcaR2.concatenateLR(l2l4_OR_R2R4_L1L2T,wtw);
		ltl=_cpcaR2.concatenateLRT(row1,row2);
		
		row1=_cpcaR2.concatenateLR(wtw,l3l4_OR_R3R4_R1R2);
		row2=_cpcaR2.concatenateLR(l3l4_OR_R3R4_R1R2T,wtw);
		rtr=_cpcaR2.concatenateLRT(row1,row2);
		
		row1=_cpcaR2.concatenateLR(l1l2_OR_R1R2_L1R1,l1l3_OR_R1R3_L1R2);
		row2=_cpcaR2.concatenateLR(l1l4_OR_R1R4_L2R1,l2l3_OR_R2R3_L2R2);
		ltr=_cpcaR2.concatenateLRT(row1,row2);
		
		row1=_cpcaR2.concatenateLR(l1l2_OR_R1R2_L1R1T,l1l3_OR_R1R3_L1R2T);
		row2=_cpcaR2.concatenateLR(l1l4_OR_R1R4_L2R1T,l2l3_OR_R2R3_L2R2T);
		rtl=_cpcaR2.concatenateLRT(row1,row2);
		
				
		(svdTC.computeSparseInverse(ltl)).zMult(ltr, auxMat5);
		(svdTC.computeSparseInverse(rtr)).zMult(rtl, auxMat2);
		auxMat5.zMult(auxMat2,auxMat3);
		phiL=svdTC.computeSVD_Tropp(auxMat3, _cpcaR2.getOmegaMatrix(auxMat3.columns()),auxMat3.columns());
		
		auxMat2.zMult(auxMat5,auxMat4);
		phiR=svdTC.computeSVD_Tropp(auxMat4, _cpcaR2.getOmegaMatrix(auxMat4.columns()),auxMat4.columns());
		
		
		
		
	}

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

