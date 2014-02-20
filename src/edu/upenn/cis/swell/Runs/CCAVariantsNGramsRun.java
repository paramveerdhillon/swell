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
import java.util.Iterator;

import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
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
		FlexCompRowMatrix xtx,xty,ytx,wtr,rtw,ltw,wtl,wtw,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T
		,L1L3_OR_R1R3_L1R2,L1L3_OR_R1R3_L1R2T,L1L4_OR_R1R4_L2R1,L1L4_OR_R1R4_L2R1T,L2L3_OR_R2R3_L2R2,L2L3_OR_R2R3_L2R2T,
		L2L4_OR_R2R4_L1L2,L2L4_OR_R2R4_L1L2T,L3L4_OR_R3R4_R1R2,L3L4_OR_R3R4_R1R2T;
		
		
		
		SVDTemplates svdTC;
		
			svdTC=new SVDTemplates(_opt,dim2);
		System.out.println("+++Generated CCA Matrices+++");
		
		
		
		
		if(_opt.typeofDecomp.equals("2viewWvsL")){
			
			xty=transform(_cpcaR2.getContextMatrix());
			ytx=transform(_cpcaR2.getContextMatrixT());
			xtx=transform(_cpcaR2.getWTWMatrix());
			L1L2_OR_R1R2_L1R1=transform(_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesv());
			L1L2_OR_R1R2_L1R1T=transform(_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesvT());
			
			if(_opt.numGrams==3){
			computeCCA2NGrams(xtx,xty,ytx,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T,svdTC,_cpcaR2,xtx.numColumns(),xty.numColumns());
			}
			
			if(_opt.numGrams==5){
				L1L3_OR_R1R3_L1R2= transform(_cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesv());
				L1L3_OR_R1R3_L1R2T =transform(_cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesvT());
				
				L1L4_OR_R1R4_L2R1= transform(_cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesv());
				L1L4_OR_R1R4_L2R1T= transform(_cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesvT());
				
				L2L3_OR_R2R3_L2R2= transform(_cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesv());
				L2L3_OR_R2R3_L2R2T= transform(_cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesvT());
				
				L2L4_OR_R2R4_L1L2= transform(_cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesv());
				L2L4_OR_R2R4_L1L2T= transform(_cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesvT());
				
				L3L4_OR_R3R4_R1R2= transform(_cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesv());
				L3L4_OR_R3R4_R1R2T= transform(_cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesvT());
				
				computeCCA2NGrams(xtx,xty,ytx,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T
						,L1L3_OR_R1R3_L1R2,L1L3_OR_R1R3_L1R2T,L1L4_OR_R1R4_L2R1,L1L4_OR_R1R4_L2R1T,L2L3_OR_R2R3_L2R2,L2L3_OR_R2R3_L2R2T,
						L2L4_OR_R2R4_L1L2,L2L4_OR_R2R4_L1L2T,L3L4_OR_R3R4_R1R2,L3L4_OR_R3R4_R1R2T,svdTC,_cpcaR2,xtx.numColumns(),xty.numColumns());
				
			}
			
			
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsR")){
			xty=transform(_cpcaR2.getContextMatrix());
			ytx=transform(_cpcaR2.getContextMatrixT());
			xtx=transform(_cpcaR2.getWTWMatrix());
			if(!_opt.depbigram){
				L1L2_OR_R1R2_L1R1=transform(_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesv());
				L1L2_OR_R1R2_L1R1T=transform(_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesvT());
			}
			else{
				L1L2_OR_R1R2_L1R1=xtx;
				L1L2_OR_R1R2_L1R1T=xtx;
			}
			
			if(_opt.numGrams==3 || _opt.numGrams==2){
				computeCCA2NGrams(xtx,xty,ytx,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T,svdTC,_cpcaR2,xtx.numColumns(),xty.numColumns());
				}
			
			if(_opt.numGrams==5){
				L1L3_OR_R1R3_L1R2= transform(_cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesv());
				L1L3_OR_R1R3_L1R2T =transform(_cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesvT());
				
				L1L4_OR_R1R4_L2R1=transform(_cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesv());
				L1L4_OR_R1R4_L2R1T=transform(_cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesvT());
				
				L2L3_OR_R2R3_L2R2=transform(_cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesv());
				L2L3_OR_R2R3_L2R2T=transform(_cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesvT());
				
				L2L4_OR_R2R4_L1L2=transform(_cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesv());
				L2L4_OR_R2R4_L1L2T=transform(_cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesvT());
				
				L3L4_OR_R3R4_R1R2=transform(_cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesv());
				L3L4_OR_R3R4_R1R2T=transform(_cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesvT());
				
				computeCCA2NGrams(xtx,xty,ytx,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T
						,L1L3_OR_R1R3_L1R2,L1L3_OR_R1R3_L1R2T,L1L4_OR_R1R4_L2R1,L1L4_OR_R1R4_L2R1T,L2L3_OR_R2R3_L2R2,L2L3_OR_R2R3_L2R2T,
						L2L4_OR_R2R4_L1L2,L2L4_OR_R2R4_L1L2T,L3L4_OR_R3R4_R1R2,L3L4_OR_R3R4_R1R2T,svdTC,_cpcaR2,xtx.numColumns(),xty.numColumns());
				
			}
			
			
			
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsLR")){
			xty=transform(_cpcaR2.getContextMatrix());
			ytx=transform(_cpcaR2.getContextMatrixT());
			xtx=transform(_cpcaR2.getWTWMatrix());
			
			System.out.println("++Got the C & W matrices++");
			
			L1L2_OR_R1R2_L1R1=transform(_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesv());
			L1L2_OR_R1R2_L1R1T=transform(_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesvT());
			
			System.out.println("++Got the matrices++");
			
			if(_opt.numGrams==3){
				computeCCA2NGrams(xtx,xty,ytx,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T,svdTC,_cpcaR2,xtx.numColumns(),xty.numColumns());
				}
			
			if(_opt.numGrams==5){
				L1L3_OR_R1R3_L1R2= transform(_cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesv());
				L1L3_OR_R1R3_L1R2T =transform(_cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesvT());
				
				L1L4_OR_R1R4_L2R1= transform(_cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesv());
				L1L4_OR_R1R4_L2R1T= transform(_cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesvT());
				
				L2L3_OR_R2R3_L2R2= transform(_cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesv());
				L2L3_OR_R2R3_L2R2T= transform(_cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesvT());
				
				L2L4_OR_R2R4_L1L2= transform(_cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesv());
				L2L4_OR_R2R4_L1L2T= transform(_cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesvT());
				
				L3L4_OR_R3R4_R1R2= transform(_cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesv());
				L3L4_OR_R3R4_R1R2T= transform(_cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesvT());
				
				computeCCA2NGrams(xtx,xty,ytx,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T
						,L1L3_OR_R1R3_L1R2,L1L3_OR_R1R3_L1R2T,L1L4_OR_R1R4_L2R1,L1L4_OR_R1R4_L2R1T,L2L3_OR_R2R3_L2R2,L2L3_OR_R2R3_L2R2T,
						L2L4_OR_R2R4_L1L2,L2L4_OR_R2R4_L1L2T,L3L4_OR_R3R4_R1R2,L3L4_OR_R3R4_R1R2T,svdTC,_cpcaR2,xtx.numColumns(),xty.numColumns());
				
			}
			
		}
		
		
		if(_opt.typeofDecomp.equals("TwoStepLRvsW")){
			
			
			wtl=transform(_cpcaR2.getWL3gramMatrix());
			wtr=transform(_cpcaR2.getWR3gramMatrix());
			ltw=transform(_cpcaR2.getWLT3gramMatrix());
			rtw=transform(_cpcaR2.getWRT3gramMatrix());
			L1L2_OR_R1R2_L1R1=transform(_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesv());
			L1L2_OR_R1R2_L1R1T=transform(_cpcaR2.getL1L2_OR_R1R2_L1R1Matrix_vTimesvT());
			wtw=transform(_cpcaR2.getWTWMatrix());
			
			
			if(_opt.numGrams==3){
				computeCCATwoStepLRvsWNGrams(wtl,wtr,ltw,rtw,wtw,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T,null,null,null,null,null,null,null,null,null,null,svdTC,_cpcaR2,wtl.numRows(),wtl.numColumns());
			}
			
			if(_opt.numGrams==5){
				wtl=transform(_cpcaR2.getWL5gramMatrix());
				wtr=transform(_cpcaR2.getWR5gramMatrix());
				ltw=transform(_cpcaR2.getWLT5gramMatrix());
				rtw=transform(_cpcaR2.getWRT5gramMatrix());
				
				
				L1L3_OR_R1R3_L1R2= transform(_cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesv());
				L1L3_OR_R1R3_L1R2T =transform(_cpcaR2.getL1L3_OR_R1R3_L1R2Matrix_vTimesvT());
				
				L1L4_OR_R1R4_L2R1= transform(_cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesv());
				L1L4_OR_R1R4_L2R1T=transform(_cpcaR2.getL1L4_OR_R1R4_L2R1Matrix_vTimesvT());
				
				L2L3_OR_R2R3_L2R2= transform(_cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesv());
				L2L3_OR_R2R3_L2R2T= transform(_cpcaR2.getL2L3_OR_R2R3_L2R2Matrix_vTimesvT());
				
				L2L4_OR_R2R4_L1L2= transform(_cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesv());
				L2L4_OR_R2R4_L1L2T= transform(_cpcaR2.getL2L4_OR_R2R4_L1L2Matrix_vTimesvT());
				
				L3L4_OR_R3R4_R1R2= transform(_cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesv());
				L3L4_OR_R3R4_R1R2T= transform(_cpcaR2.getL3L4_OR_R3R4_R1R2Matrix_vTimesvT());
				
				computeCCATwoStepLRvsWNGrams(wtl,wtr,ltw,rtw,wtw,L1L2_OR_R1R2_L1R1,L1L2_OR_R1R2_L1R1T
						,L1L3_OR_R1R3_L1R2,L1L3_OR_R1R3_L1R2T,L1L4_OR_R1R4_L2R1,L1L4_OR_R1R4_L2R1T,L2L3_OR_R2R3_L2R2,L2L3_OR_R2R3_L2R2T,
						L2L4_OR_R2R4_L1L2,L2L4_OR_R2R4_L1L2T,L3L4_OR_R3R4_R1R2,L3L4_OR_R3R4_R1R2T,svdTC,_cpcaR2,wtl.numRows(),wtl.numColumns());
				
				
			}
			
		}
	}
	

public FlexCompRowMatrix transform(FlexCompRowMatrix a){
	
	Iterator<MatrixEntry> aIt = a.iterator();
	double ent=0;
	
	while(aIt.hasNext())
		{
		MatrixEntry ment = aIt.next();
		ent =ment.get();
		if(_opt.logTrans)
			ent = Math.log(ent);
		if(_opt.sqRootTrans)
			ent = Math.sqrt(ent);
		
		a.set(ment.row(), ment.column(), ent);		
		}
	
	
return a;	
	
}


	

	private void computeCCA2NGrams(FlexCompRowMatrix xtx,
			FlexCompRowMatrix xty, FlexCompRowMatrix ytx,FlexCompRowMatrix L1L2_OR_R1R2_L1R1,FlexCompRowMatrix L1L2_OR_R1R2_L1R1T,
			SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2,int dim1,int dim2) {
		
		FlexCompRowMatrix yty=new FlexCompRowMatrix(ytx.numRows(),xty.numColumns());
		FlexCompRowMatrix auxMat5=new FlexCompRowMatrix(xtx.numColumns(),ytx.numRows());
		FlexCompRowMatrix auxMat2=new FlexCompRowMatrix(yty.numRows(),ytx.numColumns());
		FlexCompRowMatrix auxMat3=new FlexCompRowMatrix(auxMat5.numRows(),auxMat5.numRows());
		FlexCompRowMatrix auxMat4=new FlexCompRowMatrix(auxMat2.numRows(),auxMat2.numRows());
				
		
		if(_opt.numGrams==3){
			
			
			FlexCompRowMatrix topM=new FlexCompRowMatrix(xtx.numRows(),xtx.numColumns()+L1L2_OR_R1R2_L1R1.numColumns());
			FlexCompRowMatrix bottomM=new FlexCompRowMatrix(topM.numRows(),topM.numColumns());
			
			System.out.println("++Before Concatenating matrices++");
			
			topM=_cpcaR2.concatenateLR(xtx,L1L2_OR_R1R2_L1R1);
			bottomM=_cpcaR2.concatenateLR(L1L2_OR_R1R2_L1R1T,xtx);
			yty=_cpcaR2.concatenateLRT(topM,bottomM);
			
			System.out.println("++ Concatenated matrices++");
			
			if(_opt.typeofDecomp.equals("TwoStepLRvsW"))
					{
						FlexCompRowMatrix topM1=new FlexCompRowMatrix(phiLT.getRowDimension(),phiLT.getRowDimension());
						FlexCompRowMatrix bottomM1=new FlexCompRowMatrix(phiLT.getRowDimension(),phiLT.getRowDimension());
				
						FlexCompRowMatrix phiLTxtx=new FlexCompRowMatrix(phiLT.getRowDimension(),xtx.numColumns());
						FlexCompRowMatrix phiLTxtxphiL=new FlexCompRowMatrix(phiLT.getRowDimension(),phiLT.getRowDimension());
						FlexCompRowMatrix phiLTLR=new FlexCompRowMatrix(phiLT.getRowDimension(),xtx.numColumns());
						FlexCompRowMatrix phiLTLRphiR=new FlexCompRowMatrix(phiLT.getRowDimension(),phiLT.getRowDimension());
						FlexCompRowMatrix phiRTLR=new FlexCompRowMatrix(phiLT.getRowDimension(),xtx.numColumns());
						FlexCompRowMatrix phiRTLRphiL=new FlexCompRowMatrix(phiLT.getRowDimension(),phiLT.getRowDimension());
				
						phiLT=phiL.transpose();
						phiRT=phiR.transpose();
						
						MatrixFormatConversion.createDenseMatrixMTJ(phiLT).mult(xtx, phiLTxtx);
						phiLTxtx.mult(MatrixFormatConversion.createDenseMatrixMTJ(phiL),phiLTxtxphiL);
						
						MatrixFormatConversion.createDenseMatrixMTJ(phiLT).mult(L1L2_OR_R1R2_L1R1, phiLTLR);
						phiLTLR.mult(MatrixFormatConversion.createDenseMatrixMTJ(phiR),phiLTLRphiR);
						
						MatrixFormatConversion.createDenseMatrixMTJ(phiRT).mult(L1L2_OR_R1R2_L1R1T, phiRTLR);
						phiRTLR.mult(MatrixFormatConversion.createDenseMatrixMTJ(phiL),phiRTLRphiL);
						
						
						//MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(xtx, phiLTxtx);
						//phiLTxtx.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), phiLTxtxphiL);
						
						//MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(L1L2_OR_R1R2_L1R1, phiLTLR);
						//phiLTLR.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), phiLTLRphiR);
						
						//MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(L1L2_OR_R1R2_L1R1T, phiRTLR);
						//phiRTLR.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), phiRTLRphiL);
				
				
						topM1=_cpcaR2.concatenateLR(phiLTxtxphiL,phiLTLRphiR);
						bottomM1=_cpcaR2.concatenateLR(phiRTLRphiL,phiLTxtxphiL);
						
						yty=_cpcaR2.concatenateLRT(topM1,bottomM1);
					}
					
			
		}
		else{
			yty=xtx;
		}
		
		
		auxMat5=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(svdTC.computeSparseInverseSqRoot(xtx),xty);
		auxMat2=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat5,svdTC.computeSparseInverseSqRoot(yty));
		
		
		phiL=svdTC.computeSVD_Tropp(MatrixFormatConversion.createSparseMatrixCOLT(auxMat2), _cpcaR2.getOmegaMatrix(auxMat2.numColumns()),dim2);
		s=svdTC.getSingularVals();
		if(!_opt.typeofDecomp.equals("TwoStepLRvsW")){
			auxMat4=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(svdTC.computeSparseInverseSqRoot(yty),ytx);
			auxMat2=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat4,svdTC.computeSparseInverseSqRoot(xtx));
			
			phiR=svdTC.computeSVD_Tropp(MatrixFormatConversion.createSparseMatrixCOLT(auxMat2), _cpcaR2.getOmegaMatrix(auxMat2.numColumns()),dim1);
		}
		
	}
	
	
	private void computeCCA2NGrams(FlexCompRowMatrix xtx,
			FlexCompRowMatrix xty, FlexCompRowMatrix ytx,FlexCompRowMatrix L1L2_OR_R1R2_L1R1,
			FlexCompRowMatrix L1L2_OR_R1R2_L1R1T, FlexCompRowMatrix L1L3_OR_R1R3_L1R2,FlexCompRowMatrix L1L3_OR_R1R3_L1R2T,
			FlexCompRowMatrix L1L4_OR_R1R4_L2R1, FlexCompRowMatrix L1L4_OR_R1R4_L2R1T,FlexCompRowMatrix L2L3_OR_R2R3_L2R2,
			FlexCompRowMatrix L2L3_OR_R2R3_L2R2T, FlexCompRowMatrix L2L4_OR_R2R4_L1L2,FlexCompRowMatrix L2L4_OR_R2R4_L1L2T,
			FlexCompRowMatrix L3L4_OR_R3R4_R1R2, FlexCompRowMatrix L3L4_OR_R3R4_R1R2T,
			SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2,int dim1,int dim2) {
			
		FlexCompRowMatrix yty=new FlexCompRowMatrix(ytx.numRows(),xty.numColumns());
		FlexCompRowMatrix auxMat5=new FlexCompRowMatrix(xtx.numColumns(),ytx.numRows());
		FlexCompRowMatrix auxMat2=new FlexCompRowMatrix(yty.numRows(),ytx.numColumns());
		FlexCompRowMatrix auxMat3=new FlexCompRowMatrix(auxMat5.numRows(),auxMat5.numRows());
		FlexCompRowMatrix auxMat4=new FlexCompRowMatrix(auxMat2.numRows(),auxMat2.numRows());
				
		
		FlexCompRowMatrix row1=new FlexCompRowMatrix(xtx.numRows(),xtx.numColumns()+L1L2_OR_R1R2_L1R1.numColumns()+ L1L3_OR_R1R3_L1R2.numColumns()+ L1L4_OR_R1R4_L2R1.numColumns());
		FlexCompRowMatrix row2=new FlexCompRowMatrix(xtx.numRows(),xtx.numColumns()+L1L2_OR_R1R2_L1R1.numColumns()+ L1L3_OR_R1R3_L1R2.numColumns()+ L1L4_OR_R1R4_L2R1.numColumns());
		FlexCompRowMatrix row3=new FlexCompRowMatrix(xtx.numRows(),xtx.numColumns()+L1L2_OR_R1R2_L1R1.numColumns()+ L1L3_OR_R1R3_L1R2.numColumns()+ L1L4_OR_R1R4_L2R1.numColumns());
		FlexCompRowMatrix row4=new FlexCompRowMatrix(xtx.numRows(),xtx.numColumns()+L1L2_OR_R1R2_L1R1.numColumns()+ L1L3_OR_R1R3_L1R2.numColumns()+ L1L4_OR_R1R4_L2R1.numColumns());
		
		
		row1=_cpcaR2.concatenateMultiRow(xtx,L1L2_OR_R1R2_L1R1, L1L3_OR_R1R3_L1R2, L1L4_OR_R1R4_L2R1);
		row2=_cpcaR2.concatenateMultiRow(L1L2_OR_R1R2_L1R1T,xtx, L2L3_OR_R2R3_L2R2, L2L4_OR_R2R4_L1L2);
		row3=_cpcaR2.concatenateMultiRow(L1L3_OR_R1R3_L1R2T,L2L3_OR_R2R3_L2R2T ,xtx,L3L4_OR_R3R4_R1R2);
		row4=_cpcaR2.concatenateMultiRow(L1L4_OR_R1R4_L2R1T,L2L4_OR_R2R4_L1L2T,L3L4_OR_R3R4_R1R2T, xtx);
		yty=_cpcaR2.concatenateMultiCol(row1,row2,row3,row4);
		
		
		if(_opt.typeofDecomp.equals("TwoStepLRvsW"))
		{
			
			FlexCompRowMatrix row11=new FlexCompRowMatrix(phiLT.getRowDimension(),phiLT.getRowDimension());
			FlexCompRowMatrix row21=new FlexCompRowMatrix(phiLT.getRowDimension(),phiLT.getRowDimension());
			
			
			FlexCompRowMatrix aux=new FlexCompRowMatrix(phiLT.getRowDimension(),xtx.numColumns()+L2L4_OR_R2R4_L1L2.numColumns());
			
			FlexCompRowMatrix aux1=new FlexCompRowMatrix(xtx.numRows(),xtx.numColumns()+L2L4_OR_R2R4_L1L2.numColumns());
			FlexCompRowMatrix aux2=new FlexCompRowMatrix(xtx.numRows(),xtx.numColumns()+L2L4_OR_R2R4_L1L2.numColumns());
			
			FlexCompRowMatrix blockRow12Col12=new FlexCompRowMatrix(xtx.numRows()+L2L4_OR_R2R4_L1L2.numRows(),xtx.numColumns()+L2L4_OR_R2R4_L1L2.numColumns());
			FlexCompRowMatrix blockRow12Col34=new FlexCompRowMatrix(xtx.numRows()+L2L4_OR_R2R4_L1L2.numRows(),xtx.numColumns()+L2L4_OR_R2R4_L1L2.numColumns());
			FlexCompRowMatrix blockRow34Col12=new FlexCompRowMatrix(xtx.numRows()+L2L4_OR_R2R4_L1L2.numRows(),xtx.numColumns()+L2L4_OR_R2R4_L1L2.numColumns());
			FlexCompRowMatrix blockRow34Col34=new FlexCompRowMatrix(xtx.numRows()+L2L4_OR_R2R4_L1L2.numRows(),xtx.numColumns()+L2L4_OR_R2R4_L1L2.numColumns());
			FlexCompRowMatrix blockRow12Col12Phi=new FlexCompRowMatrix(phiLT.getRowDimension(),phiLT.getRowDimension());
			FlexCompRowMatrix blockRow12Col34Phi=new FlexCompRowMatrix(phiLT.getRowDimension(),phiLT.getRowDimension());
			FlexCompRowMatrix blockRow34Col12Phi=new FlexCompRowMatrix(phiLT.getRowDimension(),phiLT.getRowDimension());
			FlexCompRowMatrix  blockRow34Col34Phi=new FlexCompRowMatrix(phiLT.getRowDimension(),phiLT.getRowDimension());
		
			
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
			
			/*
			MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(blockRow12Col12, aux);
			aux.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), blockRow12Col12Phi);
			
			MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(blockRow34Col34, aux);
			aux.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), blockRow34Col34Phi);
			
			MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(blockRow34Col12, aux);
			aux.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), blockRow34Col12Phi);
			
			MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(blockRow12Col34, aux);
			aux.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), blockRow12Col34Phi);
			*/
			
			
			MatrixFormatConversion.createDenseMatrixMTJ(phiLT).mult(blockRow12Col12, aux);
			aux.mult(MatrixFormatConversion.createDenseMatrixMTJ(phiL), blockRow12Col12Phi);
			
			MatrixFormatConversion.createDenseMatrixMTJ(phiRT).mult(blockRow34Col34, aux);
			aux.mult(MatrixFormatConversion.createDenseMatrixMTJ(phiR), blockRow34Col34Phi);
			
			MatrixFormatConversion.createDenseMatrixMTJ(phiRT).mult(blockRow34Col12, aux);
			aux.mult(MatrixFormatConversion.createDenseMatrixMTJ(phiL), blockRow34Col12Phi);
			
			MatrixFormatConversion.createDenseMatrixMTJ(phiLT).mult(blockRow12Col34, aux);
			aux.mult(MatrixFormatConversion.createDenseMatrixMTJ(phiR), blockRow12Col34Phi);
			
			
	
			row11=_cpcaR2.concatenateLR(blockRow12Col12Phi,blockRow12Col34Phi);
			row21=_cpcaR2.concatenateLR(blockRow34Col12Phi,blockRow34Col34Phi);
			
			yty=_cpcaR2.concatenateLRT(row11,row21);
			

		}
		
		
		
		
		auxMat5=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(svdTC.computeSparseInverseSqRoot(xtx),xty);
		auxMat3=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat5,svdTC.computeSparseInverseSqRoot(yty));
		
		phiL=svdTC.computeSVD_Tropp(MatrixFormatConversion.createSparseMatrixCOLT(auxMat3), _cpcaR2.getOmegaMatrix(auxMat3.numColumns()),dim2);
		s=svdTC.getSingularVals();
		
		if(!_opt.typeofDecomp.equals("TwoStepLRvsW")){
		
			auxMat2=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(svdTC.computeSparseInverseSqRoot(yty),ytx);
			auxMat4=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat2, svdTC.computeSparseInverseSqRoot(xtx));
			
			
			phiR=svdTC.computeSVD_Tropp(MatrixFormatConversion.createSparseMatrixCOLT(auxMat4), _cpcaR2.getOmegaMatrix(auxMat4.numColumns()),dim1);
		}
		
	}
	
	private void computeCCA2NGramsLR(FlexCompRowMatrix ltr,
			FlexCompRowMatrix rtl, FlexCompRowMatrix wtw,
			 SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2) {
		
		
		FlexCompRowMatrix auxMat5=new FlexCompRowMatrix(wtw.numRows(),ltr.numColumns());
		FlexCompRowMatrix auxMat2=new FlexCompRowMatrix(wtw.numRows(),rtl.numColumns());
		FlexCompRowMatrix auxMat3=new FlexCompRowMatrix(auxMat5.numRows(),auxMat5.numRows());
		FlexCompRowMatrix auxMat4=new FlexCompRowMatrix(auxMat2.numRows(),auxMat2.numRows());
				
		
		auxMat5=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(svdTC.computeSparseInverseSqRoot(wtw),ltr);
		auxMat3=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat5,svdTC.computeSparseInverseSqRoot(wtw));
		
		//(svdTC.computeSparseInverse(wtw)).zMult(ltr, auxMat5);
		//(svdTC.computeSparseInverse(wtw)).zMult(rtl, auxMat2);
		//auxMat5.zMult(auxMat2,auxMat3);
		phiL=svdTC.computeSVD_Tropp(MatrixFormatConversion.createSparseMatrixCOLT(auxMat3), _cpcaR2.getOmegaMatrix(auxMat3.numColumns()),auxMat3.numColumns());
		
		auxMat2=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(svdTC.computeSparseInverseSqRoot(wtw),rtl);
		auxMat4=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat2, svdTC.computeSparseInverseSqRoot(wtw));
		
	
		phiR=svdTC.computeSVD_Tropp(MatrixFormatConversion.createSparseMatrixCOLT(auxMat4), _cpcaR2.getOmegaMatrix(auxMat4.numColumns()),auxMat4.numColumns());
		
	}
	
	private void computeCCATwoStepLRvsWNGrams(FlexCompRowMatrix wtl,
			FlexCompRowMatrix wtr, FlexCompRowMatrix ltw,
			FlexCompRowMatrix rtw,FlexCompRowMatrix wtw,
			FlexCompRowMatrix l1l2_OR_R1R2_L1R1, FlexCompRowMatrix l1l2_OR_R1R2_L1R1T, FlexCompRowMatrix l1l3_OR_R1R3_L1R2, FlexCompRowMatrix l1l3_OR_R1R3_L1R2T, FlexCompRowMatrix l1l4_OR_R1R4_L2R1, FlexCompRowMatrix l1l4_OR_R1R4_L2R1T, FlexCompRowMatrix l2l3_OR_R2R3_L2R2, FlexCompRowMatrix l2l3_OR_R2R3_L2R2T, FlexCompRowMatrix l2l4_OR_R2R4_L1L2, FlexCompRowMatrix l2l4_OR_R2R4_L1L2T, FlexCompRowMatrix l3l4_OR_R3R4_R1R2, FlexCompRowMatrix l3l4_OR_R3R4_R1R2T, SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2,int dim1,int dim2) {
		
		
		FlexCompRowMatrix wtlphiL=new FlexCompRowMatrix(wtl.numRows(),_opt.hiddenStateSize);
		FlexCompRowMatrix wtrphiR=new FlexCompRowMatrix(wtr.numRows(),_opt.hiddenStateSize);
		FlexCompRowMatrix ltwphiLT=new FlexCompRowMatrix(_opt.hiddenStateSize,wtl.numRows());
		FlexCompRowMatrix rtwphiRT=new FlexCompRowMatrix(_opt.hiddenStateSize,wtr.numRows());
		FlexCompRowMatrix wtLphiLRphiR=new FlexCompRowMatrix(wtl.numRows(),wtlphiL.numColumns()+wtrphiR.numColumns());
		FlexCompRowMatrix wtLphiLRphiRT=new FlexCompRowMatrix(wtlphiL.numColumns()+wtrphiR.numColumns(),wtl.numRows());
				
		
		if(_opt.numGrams==3){
			computeCCA2NGramsLR(l1l2_OR_R1R2_L1R1,l1l2_OR_R1R2_L1R1T,wtw,svdTC,_cpcaR2);
		}
		
		
		
		if(_opt.numGrams==5){
			computeCCA2NGramsLR(l1l2_OR_R1R2_L1R1,l1l2_OR_R1R2_L1R1T,l1l3_OR_R1R3_L1R2,l1l3_OR_R1R3_L1R2T,l1l4_OR_R1R4_L2R1,l1l4_OR_R1R4_L2R1T,l2l3_OR_R2R3_L2R2,l2l3_OR_R2R3_L2R2T,
					l2l4_OR_R2R4_L1L2,l2l4_OR_R2R4_L1L2T,l3l4_OR_R3R4_R1R2,l3l4_OR_R3R4_R1R2T,wtw,svdTC,_cpcaR2);
		}
		
		wtl.mult(MatrixFormatConversion.createDenseMatrixMTJ(phiL), wtlphiL);
		wtr.mult(MatrixFormatConversion.createDenseMatrixMTJ(phiR), wtrphiR);
		
		phiLT=phiL.transpose();
		phiRT=phiR.transpose();
		
		MatrixFormatConversion.createDenseMatrixMTJ(phiLT).mult(ltw, ltwphiLT);
		MatrixFormatConversion.createDenseMatrixMTJ(phiRT).mult(rtw, rtwphiRT);
		
		wtLphiLRphiR=_cpcaR2.concatenateLR(wtlphiL,wtrphiR);
		wtLphiLRphiRT=_cpcaR2.concatenateLRT(ltwphiLT,rtwphiRT);
		
		if(_opt.numGrams==3){
			computeCCA2NGrams( wtw,wtLphiLRphiR,wtLphiLRphiRT,l1l2_OR_R1R2_L1R1, l1l2_OR_R1R2_L1R1T,svdTC,_cpcaR2,wtLphiLRphiR.numRows(),wtLphiLRphiRT.numRows());
		}
		if(_opt.numGrams==5){
			computeCCA2NGrams(wtw,wtLphiLRphiR,wtLphiLRphiRT,l1l2_OR_R1R2_L1R1,l1l2_OR_R1R2_L1R1T
					,l1l3_OR_R1R3_L1R2,l1l3_OR_R1R3_L1R2T,l1l4_OR_R1R4_L2R1,l1l4_OR_R1R4_L2R1T,l2l3_OR_R2R3_L2R2,l2l3_OR_R2R3_L2R2T,
					l2l4_OR_R2R4_L1L2,l2l4_OR_R2R4_L1L2T,l3l4_OR_R3R4_R1R2,l3l4_OR_R3R4_R1R2T,svdTC,_cpcaR2,wtLphiLRphiR.numRows(),wtLphiLRphiRT.numRows());
			
		}
	}


/*
	private void computeCCA3NGrams(FlexCompRowMatrix view1,
			FlexCompRowMatrix view1t, FlexCompRowMatrix view2,
			FlexCompRowMatrix view2t, FlexCompRowMatrix view3,
			FlexCompRowMatrix view3t, SVDTemplates svdTC,ContextPCANGramsRepresentation _cpcaR2) {
		// TODO Auto-generated method stub
		
	}
*/

	

	private void computeCCA2NGramsLR(FlexCompRowMatrix l1l2_OR_R1R2_L1R1,
			FlexCompRowMatrix l1l2_OR_R1R2_L1R1T,
			FlexCompRowMatrix l1l3_OR_R1R3_L1R2,
			FlexCompRowMatrix l1l3_OR_R1R3_L1R2T,
			FlexCompRowMatrix l1l4_OR_R1R4_L2R1,
			FlexCompRowMatrix l1l4_OR_R1R4_L2R1T,
			FlexCompRowMatrix l2l3_OR_R2R3_L2R2,
			FlexCompRowMatrix l2l3_OR_R2R3_L2R2T,
			FlexCompRowMatrix l2l4_OR_R2R4_L1L2,
			FlexCompRowMatrix l2l4_OR_R2R4_L1L2T,
			FlexCompRowMatrix l3l4_OR_R3R4_R1R2,
			FlexCompRowMatrix l3l4_OR_R3R4_R1R2T, FlexCompRowMatrix wtw,
			SVDTemplates svdTC, ContextPCANGramsRepresentation _cpcaR2) {
		

		
		FlexCompRowMatrix rtr=new FlexCompRowMatrix(l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows(),l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows());
		FlexCompRowMatrix ltl=new FlexCompRowMatrix(l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows(),l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows());
		FlexCompRowMatrix ltr=new FlexCompRowMatrix(l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows(),l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows());
		FlexCompRowMatrix rtl=new FlexCompRowMatrix(l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows(),l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows());
		
		
		
		FlexCompRowMatrix auxMat5=new FlexCompRowMatrix(l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows(),l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows());
		FlexCompRowMatrix auxMat2=new FlexCompRowMatrix(l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows(),l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows());
		FlexCompRowMatrix auxMat3=new FlexCompRowMatrix(l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows(),l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows());
		FlexCompRowMatrix auxMat4=new FlexCompRowMatrix(l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows(),l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows());
				
		
		FlexCompRowMatrix row1=new FlexCompRowMatrix(l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows(),l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows());
		FlexCompRowMatrix row2=new FlexCompRowMatrix(l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows(),l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows());
		
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
		
		
		auxMat5=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(svdTC.computeSparseInverseSqRoot(ltl),ltr);
		auxMat3=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat5,svdTC.computeSparseInverseSqRoot(rtr));
		
		
		phiL=svdTC.computeSVD_Tropp(MatrixFormatConversion.createSparseMatrixCOLT(auxMat3), _cpcaR2.getOmegaMatrix(auxMat3.numColumns()),auxMat3.numColumns());
		
		auxMat2=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(svdTC.computeSparseInverseSqRoot(rtr),rtl);
		auxMat4=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat2, svdTC.computeSparseInverseSqRoot(ltl));
		
		
		phiR=svdTC.computeSVD_Tropp(MatrixFormatConversion.createSparseMatrixCOLT(auxMat4), _cpcaR2.getOmegaMatrix(auxMat4.numColumns()),auxMat4.numColumns());
		
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

