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

import jeigen.DenseMatrix;
import jeigen.DenseMatrix.SvdResult;
//import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.NotConvergedException;
import no.uib.cipr.matrix.SVD;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import static jeigen.Shortcuts.*;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import Jama.Matrix;
import Jama.SingularValueDecomposition;
import edu.upenn.cis.swell.IO.Options;
import edu.upenn.cis.swell.MathUtils.CenterScaleNormalizeUtils;
import edu.upenn.cis.swell.MathUtils.MatrixFormatConversion;
import edu.upenn.cis.swell.MathUtils.SVDTemplates;
import edu.upenn.cis.swell.SpectralRepresentations.ContextPCARepresentation;

public class CCAVariantsRun implements Serializable {

	private Options _opt;
	private ContextPCARepresentation _cpcaR; 
	private int dim2=0;
	static final long serialVersionUID = 42L;
	Matrix phiL, phiR, phiLT,phiRT,phiLCSU, phiRCSU,phiL_1stage,phiR_1stage ;
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
		if(_opt.logTrans || _opt.sqRootTrans)
			_cpcaR2.transformMatrices();
		
		SVDTemplates svdTC;
		
			svdTC=new SVDTemplates(_opt,dim2);

		System.out.println("+++Generated CCA Matrices+++");
		
		if(_opt.typeofDecomp.equals("2viewWvsL")){
			
			computeCCA2(_cpcaR2.getWTLMatrix(),_cpcaR2.getLTWMatrix(),_cpcaR2.getLTLMatrix(),_cpcaR2.getWTWMatrix(),svdTC,_cpcaR2);
		}
		
		
		if(_opt.typeofDecomp.equals("2viewWvsR")){
			computeCCA2(_cpcaR2.getWTRMatrix(),_cpcaR2.getRTWMatrix(),_cpcaR2.getRTRMatrix(),_cpcaR2.getWTWMatrix(),svdTC,_cpcaR2);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsLR")){
			computeCCA2(_cpcaR2.getWTLRMatrix(), _cpcaR2.getLRTWMatrix(),_cpcaR2.getLRTLRMatrix(),_cpcaR2.getWTWMatrix(),svdTC,_cpcaR2);
		}
		
		//3 view is not implemented as yet.
		if(_opt.typeofDecomp.equals("3viewLvsWvsR") || _opt.typeofDecomp.equals("TwoStepLRvsW") || _opt.typeofDecomp.equals("LRMVLVariant2")){
		
			if(_opt.typeofDecomp.equals("TwoStepLRvsW"))
				computeCCATwoStepLRvsW(_cpcaR2.getLTRMatrix(),_cpcaR2.getRTLMatrix(),_cpcaR2.getLTLMatrix(),_cpcaR2.getRTRMatrix(),_cpcaR2.getWTWMatrix(),
						_cpcaR2.getWTLMatrix(),_cpcaR2.getWTRMatrix(),
						_cpcaR2.getLTWMatrix(),_cpcaR2.getRTWMatrix(),svdTC,_cpcaR2);
			
			if(_opt.typeofDecomp.equals("LRMVLVariant2"))
				computeCCALRMVL2(_cpcaR2.getLTRMatrix(),_cpcaR2.getRTLMatrix(),_cpcaR2.getLTLMatrix(),_cpcaR2.getRTRMatrix(),_cpcaR2.getWTWMatrix(),
						_cpcaR2.getWTLMatrix(),_cpcaR2.getWTRMatrix(),
						_cpcaR2.getLTWMatrix(),_cpcaR2.getRTWMatrix(),svdTC,_cpcaR2);
			
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
		DenseDoubleMatrix2D phiLCOLT=new DenseDoubleMatrix2D(xtx.rows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D phiRCOLT=new DenseDoubleMatrix2D(yty.rows(),_opt.hiddenStateSize);
		
		DenseDoubleMatrix2D auxMat1=new DenseDoubleMatrix2D(xtx.rows(),xty.columns());
		DenseDoubleMatrix2D auxMat2=new DenseDoubleMatrix2D(yty.rows(),ytx.columns());
		DenseDoubleMatrix2D auxMat3=new DenseDoubleMatrix2D(auxMat1.rows(),auxMat1.columns());
		DenseDoubleMatrix2D auxMat4=new DenseDoubleMatrix2D(auxMat2.rows(),auxMat2.columns());
				
		int dim1=xty.rows();
		int dim2=xty.columns();
		
		System.out.println("+++Initialized auxiliary matrices+++");
		
		auxMat1=MatrixFormatConversion.multiplySparseDenseScaleRow(svdTC.computeSparseInverseSqRoot(xtx), xty);
		
				
		System.out.println("+++Computed Sparse Inverses+++");
		
		auxMat1.zMult(svdTC.computeDenseInverseSqRoot(yty),auxMat3);	
		
		
		System.out.println("+++Entering SVD computation+++");
		phiLCSU=svdTC.computeSVD_Tropp(auxMat3, _cpcaR2.getOmegaMatrix(auxMat3.columns()),dim2);
		s=svdTC.getSingularVals();
		(svdTC.computeSparseInverseSqRoot(xtx)).zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiLCSU), phiLCOLT);
		phiL=MatrixFormatConversion.createDenseMatrixJAMA(phiLCOLT);
				
				
		svdTC.computeDenseInverseSqRoot(yty).zMult(ytx,auxMat2);
		
		auxMat4=MatrixFormatConversion.multiplySparseDenseScaleCol(auxMat2, svdTC.computeSparseInverseSqRoot(xtx));
		
		phiRCSU=svdTC.computeSVD_Tropp(auxMat4, _cpcaR2.getOmegaMatrix(auxMat4.columns()),dim1);
		s=svdTC.getSingularVals();
		svdTC.computeDenseInverseSqRoot(yty).zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiRCSU), phiRCOLT);
		phiR=MatrixFormatConversion.createDenseMatrixJAMA(phiRCOLT);
		
	}
	
	private void computeCCA2(DenseDoubleMatrix2D xty,
			DenseDoubleMatrix2D ytx, DenseDoubleMatrix2D yty,
			DenseDoubleMatrix2D xtx, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2) {
		
		
		System.out.println("+++Entering CCA Compute Function+++");
		
		DenseDoubleMatrix2D phiLCOLT=new DenseDoubleMatrix2D(xtx.rows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D phiRCOLT=new DenseDoubleMatrix2D(yty.rows(),_opt.hiddenStateSize);
		
		
		DenseDoubleMatrix2D auxMat1=new DenseDoubleMatrix2D(xtx.rows(),xty.columns());
		DenseDoubleMatrix2D auxMat2=new DenseDoubleMatrix2D(yty.rows(),ytx.columns());
		DenseDoubleMatrix2D auxMat3=new DenseDoubleMatrix2D(auxMat1.rows(),auxMat1.rows());
		DenseDoubleMatrix2D auxMat4=new DenseDoubleMatrix2D(auxMat2.rows(),auxMat2.rows());
		DenseDoubleMatrix2D xtxU,xtxS,xtxV,ytyU,ytyS,ytyV,xtxSNew,ytySNew;
		
		DenseDoubleMatrix2D auxM1=new DenseDoubleMatrix2D(xtx.rows(),xtx.columns());
		DenseDoubleMatrix2D auxM2=new DenseDoubleMatrix2D(xtx.rows(),xtx.columns());
		DenseDoubleMatrix2D auxM3=new DenseDoubleMatrix2D(yty.rows(),yty.columns());
		DenseDoubleMatrix2D auxM4=new DenseDoubleMatrix2D(yty.rows(),yty.columns());
		
		int dim1=xty.rows();
		int dim2=xty.columns();
		
		DenseDoubleAlgebra dalg=new DenseDoubleAlgebra();
		
		System.out.println("+++Initialized auxiliary matrices+++");
		
		 xtxU= (DenseDoubleMatrix2D) dalg.svd(xtx).getU();
		 xtxS= (DenseDoubleMatrix2D) dalg.svd(xtx).getS();
		 xtxV= (DenseDoubleMatrix2D) dalg.svd(xtx).getV();
	
		  xtxSNew=new DenseDoubleMatrix2D( xtxS.columns(), xtxS.columns());
		 
		 for(int j=0;j < xtxS.columns();j++){
				 double ent = xtxS.get(j,j);
				 ent =1/Math.sqrt(ent);
				 xtxSNew.set(j, j, ent);
		 }
		 
		 xtxU.zMult(xtxSNew, auxM1);
		 auxM1.zMult(xtxV, auxM2);
		 
		 ytyU= (DenseDoubleMatrix2D) dalg.svd(yty).getU();
		 ytyS= (DenseDoubleMatrix2D) dalg.svd(yty).getS();
		 ytyV= (DenseDoubleMatrix2D) dalg.svd(yty).getV();
		 
		 ytySNew=new DenseDoubleMatrix2D(ytyS.columns(), ytyS.columns());
		
		 for(int j=0;j < ytyS.columns();j++){
			 double ent = ytyS.get(j,j);
			 ent =1/Math.sqrt(ent);
			 ytySNew.set(j, j, ent);
		 }
	 
		 ytyU.zMult(ytySNew, auxM3);
		 auxM3.zMult(ytyV, auxM4);
		 
		 
		((DenseDoubleMatrix2D)auxM2).zMult(xty, auxMat1);
		
		System.out.println("+++Computed 1 inverse+++");
		
		
		
		((DenseDoubleMatrix2D) auxM4).zMult(ytx, auxMat2);
		
		System.out.println("+++Computed Inverses+++");
		auxMat1.zMult(auxM4,auxMat3);
		
		System.out.println("+++Entering SVD computation+++");
		phiLCSU=svdTC.computeSVD_Tropp(auxMat3, _cpcaR2.getOmegaMatrix(auxMat3.columns()),dim1);
		s=svdTC.getSingularVals();
		auxM2.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiLCSU), phiLCOLT);
		phiL=MatrixFormatConversion.createDenseMatrixJAMA(phiLCOLT);
		
		auxMat2.zMult(auxM2,auxMat4);
		phiRCSU=svdTC.computeSVD_Tropp(auxMat4, _cpcaR2.getOmegaMatrix(auxMat4.columns()),dim2);
		s=svdTC.getSingularVals();
		auxM4.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiRCSU), phiRCOLT);
		phiR=MatrixFormatConversion.createDenseMatrixJAMA(phiRCOLT);
		
		
		
	}
	
	
	
	private void computeCCA2(FlexCompRowMatrix xty,
			FlexCompRowMatrix ytx, FlexCompRowMatrix yty,
			FlexCompRowMatrix xtx, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2) {
		
		CenterScaleNormalizeUtils csu =new CenterScaleNormalizeUtils(_opt);
		System.out.println("+++Entering CCA Compute Function+++");
		DenseDoubleMatrix2D phiLCOLT=new DenseDoubleMatrix2D(xtx.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D phiRCOLT=new DenseDoubleMatrix2D(yty.numRows(),_opt.hiddenStateSize);
		
		
		FlexCompRowMatrix auxMat1=new FlexCompRowMatrix(xtx.numRows(),xty.numColumns());
		FlexCompRowMatrix auxMat2=new FlexCompRowMatrix(yty.numRows(),ytx.numColumns());
		FlexCompRowMatrix auxMat3=new FlexCompRowMatrix(auxMat1.numRows(),auxMat1.numColumns());
		FlexCompRowMatrix auxMat4=new FlexCompRowMatrix(auxMat2.numRows(),auxMat2.numColumns());
				
		//FlexCompRowMatrix auxMat5=new FlexCompRowMatrix(auxMat3.numRows(),xty.numRows());
		//FlexCompRowMatrix auxMat6=new FlexCompRowMatrix(auxMat4.numRows(),xty.numRows());
		
		
		int dim1=ytx.numRows();
		int dim2=xty.numRows();
		
		System.out.println("+++Initialized auxiliary matrices+++");
				
		//(svdTC.computeSparseInverse(xtx)).zMult(xty, auxMat1);
		
		auxMat1=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(svdTC.computeSparseInverseSqRoot(xtx),xty);
		
		auxMat3=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat1,svdTC.computeSparseInverseSqRoot(yty));
		
		
		System.out.println("+++Computed 1 inverse+++");
		
		//(svdTC.computeSparseInverse(yty)).zMult(ytx, auxMat2);
		
		auxMat2=MatrixFormatConversion.multLargeSparseMatricesJEIGEN((svdTC.computeSparseInverseSqRoot(yty)),ytx);
		
		auxMat4=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat2,svdTC.computeSparseInverseSqRoot(xtx));
		
		
		System.out.println("+++Computed Inverses+++");
		
		//auxMat1.zMult(auxMat2,auxMat3);
		
		System.out.println("+++Entering SVD computation+++");
		
		
		phiLCSU=svdTC.computeSVD_Tropp(MatrixFormatConversion.createSparseMatrixCOLT(auxMat3), _cpcaR2.getOmegaMatrix(auxMat3.numColumns()),dim1);
		s=svdTC.getSingularVals();
		
		//phiL=phiLCSU; 
		
		MatrixFormatConversion.createSparseMatrixCOLT((svdTC.computeSparseInverseSqRoot(xtx))).zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiLCSU), phiLCOLT);
		phiL=MatrixFormatConversion.createDenseMatrixJAMA(phiLCOLT);
		
		phiRCSU=svdTC.computeSVD_Tropp(MatrixFormatConversion.createSparseMatrixCOLT(auxMat4), _cpcaR2.getOmegaMatrix(auxMat4.numColumns()),dim2);
		
		MatrixFormatConversion.createSparseMatrixCOLT((svdTC.computeSparseInverseSqRoot(yty))).zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiRCSU), phiRCOLT);
		phiR=MatrixFormatConversion.createDenseMatrixJAMA(phiRCOLT);
		
		
		//phiR=phiRCSU;
	}
	
	private void computeCCA2LRMVL2(DenseDoubleMatrix2D U, DenseDoubleMatrix2D UT, DenseDoubleMatrix2D UminusHalf,DenseDoubleMatrix2D UTminusHalf,
			FlexCompRowMatrix xty,
			FlexCompRowMatrix ytx, FlexCompRowMatrix yty,
			FlexCompRowMatrix xtx, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2) {
		
		CenterScaleNormalizeUtils csu =new CenterScaleNormalizeUtils(_opt);
		System.out.println("+++Entering CCA Compute Function+++");
		DenseDoubleMatrix2D phiLCOLT=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		DenseDoubleMatrix2D phiRCOLT=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		
		
		DenseDoubleMatrix2D a1Mat=new DenseDoubleMatrix2D(UT.rows(),xtx.numColumns());
		DenseDoubleMatrix2D a2Mat=new DenseDoubleMatrix2D(UT.rows(),UminusHalf.columns());
		DenseDoubleMatrix2D a3Mat=new DenseDoubleMatrix2D(UT.rows(),UT.columns());
		DenseDoubleMatrix2D a4Mat=new DenseDoubleMatrix2D(UT.rows(),xty.numColumns());
		DenseDoubleMatrix2D a5Mat=new DenseDoubleMatrix2D(UT.rows(),U.columns());
		DenseDoubleMatrix2D a6Mat=new DenseDoubleMatrix2D(UT.rows(),UTminusHalf.columns());
		DenseDoubleMatrix2D a7Mat=new DenseDoubleMatrix2D(UT.rows(),yty.numColumns());
		DenseDoubleMatrix2D a8Mat=new DenseDoubleMatrix2D(UT.rows(),UT.rows());
		
		
		DenseDoubleMatrix2D a11Mat=new DenseDoubleMatrix2D(UT.rows(),yty.numColumns());
		DenseDoubleMatrix2D a21Mat=new DenseDoubleMatrix2D(UT.rows(),UminusHalf.columns());
		DenseDoubleMatrix2D a31Mat=new DenseDoubleMatrix2D(UT.rows(),UT.columns());
		DenseDoubleMatrix2D a41Mat=new DenseDoubleMatrix2D(UT.rows(),ytx.numColumns());
		DenseDoubleMatrix2D a51Mat=new DenseDoubleMatrix2D(UT.rows(),U.columns());
		DenseDoubleMatrix2D a61Mat=new DenseDoubleMatrix2D(UT.rows(),UTminusHalf.columns());
		DenseDoubleMatrix2D a71Mat=new DenseDoubleMatrix2D(UT.rows(),xtx.numColumns());
		DenseDoubleMatrix2D a81Mat=new DenseDoubleMatrix2D(UT.rows(),UT.rows());
		
		
		int dim1=ytx.numRows();
		int dim2=xty.numRows();
		
		System.out.println("+++Initialized auxiliary matrices+++");
				
		//(svdTC.computeSparseInverse(xtx)).zMult(xty, auxMat1);
		
		UTminusHalf.zMult(MatrixFormatConversion.createSparseMatrixCOLT(svdTC.computeSparseInverseSqRoot(xtx)), a1Mat);
		a1Mat.zMult(UminusHalf, a2Mat);
		a2Mat.zMult(UT, a3Mat);
		a3Mat.zMult(MatrixFormatConversion.createSparseMatrixCOLT(xty), a4Mat);
		a4Mat.zMult(U, a5Mat);
		a5Mat.zMult(UTminusHalf, a6Mat);
		a6Mat.zMult(MatrixFormatConversion.createSparseMatrixCOLT(svdTC.computeSparseInverseSqRoot(yty)), a7Mat);
		a7Mat.zMult(UminusHalf, a8Mat);
		
		
		UTminusHalf.zMult(MatrixFormatConversion.createSparseMatrixCOLT(svdTC.computeSparseInverseSqRoot(yty)), a11Mat);
		a11Mat.zMult(UminusHalf, a21Mat);
		a21Mat.zMult(UT, a31Mat);
		a31Mat.zMult(MatrixFormatConversion.createSparseMatrixCOLT(ytx), a41Mat);
		a41Mat.zMult(U, a51Mat);
		a51Mat.zMult(UTminusHalf, a61Mat);
		a61Mat.zMult(MatrixFormatConversion.createSparseMatrixCOLT(svdTC.computeSparseInverseSqRoot(xtx)), a71Mat);
		a71Mat.zMult(UminusHalf, a81Mat);
		
		
		
		
		System.out.println("+++Computed Inverses+++");
		
		//auxMat1.zMult(auxMat2,auxMat3);
		
		System.out.println("+++Entering SVD computation+++");
		
		
		phiLCSU=MatrixFormatConversion.createDenseMatrixJAMA(a8Mat).svd().getU();
		s=svdTC.getSingularVals();
		
		//phiL= csu.center_and_scale(phiLCSU);
		//phiL=phiLCSU commented 3/5/14
		//phiL= MatrixFormatConversion.multLargeSparseMatricesJEIGEN((svdTC.computeSparseInverseSqRoot(xtx)),phiLCSU);
		
		a2Mat.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiLCSU), phiLCOLT);
		phiL=MatrixFormatConversion.createDenseMatrixJAMA(phiLCOLT);
		
		phiRCSU=MatrixFormatConversion.createDenseMatrixJAMA(a81Mat).svd().getU();
		
		a21Mat.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiRCSU), phiRCOLT);
		phiR=MatrixFormatConversion.createDenseMatrixJAMA(phiRCOLT);
		
		
		//phiR=csu.center_and_scale(phiRCSU);
		//phiR=phiRCSU;
	}
	
	
	

	private void computeCCA2Old(FlexCompRowMatrix xty,
			FlexCompRowMatrix ytx, FlexCompRowMatrix yty,
			FlexCompRowMatrix xtx, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2) {
		
		
		System.out.println("+++Entering CCA Compute Function+++");
		
		FlexCompRowMatrix auxMat1=new FlexCompRowMatrix(xtx.numRows(),xty.numColumns());
		FlexCompRowMatrix auxMat2=new FlexCompRowMatrix(yty.numRows(),ytx.numColumns());
		FlexCompRowMatrix auxMat3=new FlexCompRowMatrix(auxMat1.numRows(),auxMat1.numColumns());
		FlexCompRowMatrix auxMat4=new FlexCompRowMatrix(auxMat2.numRows(),auxMat2.numColumns());
				
		FlexCompRowMatrix auxMat5=new FlexCompRowMatrix(auxMat3.numRows(),xty.numRows());
		FlexCompRowMatrix auxMat6=new FlexCompRowMatrix(auxMat4.numRows(),xty.numRows());
		
		
		int dim1=ytx.numColumns();
		int dim2=xty.numColumns();
		
		System.out.println("+++Initialized auxiliary matrices+++");
				
		//(svdTC.computeSparseInverse(xtx)).zMult(xty, auxMat1);
		
		auxMat1=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(svdTC.computeSparseInverse(xtx),xty);
		
		auxMat3=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat1,svdTC.computeSparseInverse(yty));
		
		auxMat5=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat3,ytx);
		
		System.out.println("+++Computed 1 inverse+++");
		
		//(svdTC.computeSparseInverse(yty)).zMult(ytx, auxMat2);
		
		auxMat2=MatrixFormatConversion.multLargeSparseMatricesJEIGEN((svdTC.computeSparseInverse(yty)),ytx);
		
		auxMat4=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat2,svdTC.computeSparseInverse(xtx));
		
		auxMat6=MatrixFormatConversion.multLargeSparseMatricesJEIGEN(auxMat4,xty);
		
		System.out.println("+++Computed Inverses+++");
		
		//auxMat1.zMult(auxMat2,auxMat3);
		
		System.out.println("+++Entering SVD computation+++");
		
		
		phiL=svdTC.computeSVD_Tropp(MatrixFormatConversion.createSparseMatrixCOLT(auxMat5), _cpcaR2.getOmegaMatrix(auxMat5.numColumns()),dim1);
		s=svdTC.getSingularVals();
		
		phiR=svdTC.computeSVD_Tropp(MatrixFormatConversion.createSparseMatrixCOLT(auxMat6), _cpcaR2.getOmegaMatrix(auxMat6.numColumns()),dim2);
		
	}
	
	
	private void computeCCATwoStepLRvsW(FlexCompRowMatrix ltr,
			FlexCompRowMatrix rtl, FlexCompRowMatrix ltl,
			FlexCompRowMatrix rtr, FlexCompRowMatrix wtw,
			FlexCompRowMatrix wtl, FlexCompRowMatrix wtr, FlexCompRowMatrix ltw, FlexCompRowMatrix rtw, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2) {
		
		
		
		DenseDoubleMatrix2D WTLphiL=new DenseDoubleMatrix2D(wtl.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D WTRphiR=new DenseDoubleMatrix2D(wtr.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D LTWphiL=new DenseDoubleMatrix2D(_opt.hiddenStateSize,wtl.numRows());
		DenseDoubleMatrix2D RTWphiR=new DenseDoubleMatrix2D(_opt.hiddenStateSize,wtr.numRows());
		DenseDoubleMatrix2D WTLphiLWTRphiR=new DenseDoubleMatrix2D(wtl.numRows(),2*_opt.hiddenStateSize);
		DenseDoubleMatrix2D phiLTLTWphiRTRTW=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize,wtl.numRows());
		
		DenseDoubleMatrix2D LRTLRphiLphiR=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize,2*_opt.hiddenStateSize);
				
		DenseDoubleMatrix2D phiLT_LTL_phiL=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		DenseDoubleMatrix2D phiLT_LTR_phiR=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		DenseDoubleMatrix2D phiRT_RTL_phiL=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		DenseDoubleMatrix2D phiRT_RTR_phiR=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		
		DenseDoubleMatrix2D  ltl_phiL=new DenseDoubleMatrix2D(ltl.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D  ltr_phiR=new DenseDoubleMatrix2D(ltr.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D  rtl_phiL=new DenseDoubleMatrix2D(rtl.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D  rtr_phiR=new DenseDoubleMatrix2D(rtr.numRows(),_opt.hiddenStateSize);
		
		
		computeCCA2(ltr,rtl, rtr,ltl,svdTC,_cpcaR2);
		
		MatrixFormatConversion.createSparseMatrixCOLT(wtl).zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), WTLphiL);
		MatrixFormatConversion.createSparseMatrixCOLT(wtr).zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), WTRphiR);
		
		phiL_1stage=phiL;
		phiR_1stage=phiR;

		phiLT=phiL.transpose();
		phiRT=phiR.transpose();
		
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(MatrixFormatConversion.createSparseMatrixCOLT(ltw), LTWphiL);
		MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(MatrixFormatConversion.createSparseMatrixCOLT(rtw), RTWphiR);
		
		WTLphiLWTRphiR=_cpcaR2.concatenateLR(WTLphiL,WTRphiR);
		phiLTLTWphiRTRTW=_cpcaR2.concatenateLRT(LTWphiL,RTWphiR);
		
		MatrixFormatConversion.createSparseMatrixCOLT(ltl).zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), ltl_phiL);
		MatrixFormatConversion.createSparseMatrixCOLT(ltr).zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), ltr_phiR);
		MatrixFormatConversion.createSparseMatrixCOLT(rtl).zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), rtl_phiL);
		MatrixFormatConversion.createSparseMatrixCOLT(rtr).zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), rtr_phiR);
		
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(ltl_phiL, phiLT_LTL_phiL);
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(ltr_phiR, phiLT_LTR_phiR);
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(rtl_phiL, phiRT_RTL_phiL);
		MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(rtr_phiR, phiRT_RTR_phiR);
		
		LRTLRphiLphiR=_cpcaR2.concatenateLRT(_cpcaR2.concatenateLR(phiLT_LTL_phiL,phiLT_LTR_phiR),_cpcaR2.concatenateLR(phiRT_RTL_phiL,phiRT_RTR_phiR));
		
		computeCCA2(WTLphiLWTRphiR,phiLTLTWphiRTRTW, LRTLRphiLphiR,MatrixFormatConversion.createSparseMatrixCOLT(wtw),svdTC,_cpcaR2);
		
	}

	
	private void computeCCALRMVL2(FlexCompRowMatrix ltr,
			FlexCompRowMatrix rtl, FlexCompRowMatrix ltl,
			FlexCompRowMatrix rtr, FlexCompRowMatrix wtw,
			FlexCompRowMatrix wtl, FlexCompRowMatrix wtr, FlexCompRowMatrix ltw, FlexCompRowMatrix rtw, SVDTemplates svdTC,ContextPCARepresentation _cpcaR2) {
		
		CenterScaleNormalizeUtils mathUtils=new CenterScaleNormalizeUtils(_opt);
		
		Matrix UHat= _cpcaR2.initializeRandomly(ltr.numRows());
		DenseDoubleMatrix2D UHatT = MatrixFormatConversion.createDenseMatrixCOLT(UHat.transpose());
		DenseDoubleMatrix2D UHatminusHalf= new DenseDoubleMatrix2D(UHat.getRowDimension(),UHat.getColumnDimension());
		DenseDoubleMatrix2D UHatTminusHalf= new DenseDoubleMatrix2D(UHat.getColumnDimension(),UHat.getRowDimension());
	
		
		UHatminusHalf=getUHatMinusHalfSVD(MatrixFormatConversion.createDenseMatrixCOLT(UHat));
		UHatTminusHalf=getUHatMinusHalfSVD(UHatT);
		
			
		
		DenseDoubleMatrix2D WTLphiL=new DenseDoubleMatrix2D(wtl.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D WTRphiR=new DenseDoubleMatrix2D(wtr.numRows(),_opt.hiddenStateSize);
		
		DenseDoubleMatrix2D WTLUHat=new DenseDoubleMatrix2D(wtl.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D WTRUHat=new DenseDoubleMatrix2D(wtr.numRows(),_opt.hiddenStateSize);
		
		DenseDoubleMatrix2D LTWphiL=new DenseDoubleMatrix2D(_opt.hiddenStateSize,wtl.numRows());
		DenseDoubleMatrix2D RTWphiR=new DenseDoubleMatrix2D(_opt.hiddenStateSize,wtr.numRows());
		
		DenseDoubleMatrix2D UHatTLTW=new DenseDoubleMatrix2D(_opt.hiddenStateSize,wtl.numColumns());
		DenseDoubleMatrix2D UHatTRTW=new DenseDoubleMatrix2D(_opt.hiddenStateSize,wtr.numColumns());
		
		
		
		DenseDoubleMatrix2D WTLphiLWTRphiR=new DenseDoubleMatrix2D(wtl.numRows(),2*_opt.hiddenStateSize);
		DenseDoubleMatrix2D phiLTLTWphiRTRTW=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize,wtl.numRows());
		
		DenseDoubleMatrix2D LRTLRphiLphiR=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize,2*_opt.hiddenStateSize);
				
		DenseDoubleMatrix2D UHatTltlUHat_phiL=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		DenseDoubleMatrix2D UHatTltrUHat_phiR=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		DenseDoubleMatrix2D UHatTrtlUHat_phiL=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		DenseDoubleMatrix2D UHatTrtrUHat_phiR=new DenseDoubleMatrix2D(_opt.hiddenStateSize,_opt.hiddenStateSize);
		
		
		DenseDoubleMatrix2D ltlUHat=new DenseDoubleMatrix2D(ltl.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D ltrUHat=new DenseDoubleMatrix2D(ltr.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D rtlUHat=new DenseDoubleMatrix2D(rtl.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D rtrUHat=new DenseDoubleMatrix2D(rtr.numRows(),_opt.hiddenStateSize);
		
		DenseDoubleMatrix2D ltlUHat_phiL=new DenseDoubleMatrix2D(ltl.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D ltrUHat_phiR=new DenseDoubleMatrix2D(ltr.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D rtlUHat_phiL=new DenseDoubleMatrix2D(rtl.numRows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D rtrUHat_phiR=new DenseDoubleMatrix2D(rtr.numRows(),_opt.hiddenStateSize);
		
		System.out.println("Beginning Frob. Norm: "+UHat.normF());
		
		for(int iter=0; iter<10; iter++){
			computeCCA2LRMVL2(MatrixFormatConversion.createDenseMatrixCOLT(UHat), UHatT,UHatminusHalf,UHatTminusHalf, ltr,rtl, rtr,ltl,svdTC,_cpcaR2);

			MatrixFormatConversion.createSparseMatrixCOLT(wtl).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat), WTLUHat);
			MatrixFormatConversion.createSparseMatrixCOLT(wtr).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat), WTRUHat);

			WTLUHat.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), WTLphiL);
			WTRUHat.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), WTRphiR);

			
			phiLT=phiL.transpose();
			phiRT=phiR.transpose();
			
			phiL_1stage=phiL;
			phiR_1stage=phiR;

			MatrixFormatConversion.createDenseMatrixCOLT(phiLT).zMult(UHatT, UHatTLTW);
			MatrixFormatConversion.createDenseMatrixCOLT(phiRT).zMult(UHatT, UHatTRTW);

			UHatTLTW.zMult(MatrixFormatConversion.createSparseMatrixCOLT(ltw), LTWphiL);
			UHatTRTW.zMult(MatrixFormatConversion.createSparseMatrixCOLT(rtw), RTWphiR);

			
			WTLphiLWTRphiR=_cpcaR2.concatenateLR(WTLphiL,WTRphiR);
			phiLTLTWphiRTRTW=_cpcaR2.concatenateLRT(LTWphiL,RTWphiR);

			MatrixFormatConversion.createSparseMatrixCOLT(ltl).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat), ltlUHat);
			MatrixFormatConversion.createSparseMatrixCOLT(ltr).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat), ltrUHat);
			MatrixFormatConversion.createSparseMatrixCOLT(rtl).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat), rtlUHat);
			MatrixFormatConversion.createSparseMatrixCOLT(rtr).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat), rtrUHat);

			ltlUHat.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), ltlUHat_phiL);
			ltrUHat.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), ltrUHat_phiR);
			rtlUHat.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiL), rtlUHat_phiL);
			rtrUHat.zMult(MatrixFormatConversion.createDenseMatrixCOLT(phiR), rtrUHat_phiR);

			UHatT.zMult(ltlUHat_phiL, UHatTltlUHat_phiL);
			UHatT.zMult(ltrUHat_phiR, UHatTltrUHat_phiR);
			UHatT.zMult(rtlUHat_phiL, UHatTrtlUHat_phiL);
			UHatT.zMult(rtrUHat_phiR, UHatTrtrUHat_phiR);

			LRTLRphiLphiR=_cpcaR2.concatenateLRT(_cpcaR2.concatenateLR(UHatTltlUHat_phiL,UHatTltrUHat_phiR),_cpcaR2.concatenateLR(UHatTrtlUHat_phiL,UHatTrtrUHat_phiR));

			computeCCA2(WTLphiLWTRphiR,phiLTLTWphiRTRTW, LRTLRphiLphiR,MatrixFormatConversion.createSparseMatrixCOLT(wtw),svdTC,_cpcaR2);
		
		
			UHat=MatrixFormatConversion.createDenseMatrixJAMA(_cpcaR2.concatenateLRT(MatrixFormatConversion.createDenseMatrixCOLT(phiL),MatrixFormatConversion.createDenseMatrixCOLT(phiL)));
			UHat = mathUtils.normalize(UHat);
			
			UHatT =MatrixFormatConversion.createDenseMatrixCOLT(UHat.transpose());
			UHatminusHalf=getUHatMinusHalfSVD(MatrixFormatConversion.createDenseMatrixCOLT(UHat));
			UHatTminusHalf=getUHatMinusHalfSVD(UHatT);
			System.out.println("Frob. Norm at iteration "+iter+": "+UHat.normF());
		
		
		}
	}
	
	
	
	
	private DenseDoubleMatrix2D getUHatMinusHalfSVD(DenseDoubleMatrix2D uHatT) {
		
		
		
		DenseMatrix dm1=MatrixFormatConversion.createDenseMatrixJEIGEN(uHatT);
		
		SvdResult result =  dm1.svd();
		 
		DenseMatrix U= result.U;
		DenseMatrix S=result.S;
		DenseMatrix V=result.V;
		
		DenseMatrix SNew=new DenseMatrix(U.cols,V.cols);
		
		DenseMatrix USNew= new DenseMatrix(U.rows,SNew.cols);
		DenseMatrix USVNew= new DenseMatrix(USNew.rows,V.rows);
		
		
		for(int i=0; i< S.rows;i++){
			SNew.set(i, i, 1/Math.sqrt(S.get(i, 0)));
		}
		
		USNew=U.mmul(SNew);
		USVNew=USNew.mmul(V.t());
		
		return MatrixFormatConversion.createDenseMatrixCOLT(USVNew);
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
		
		String eigDictL=_opt.serializeRun+"EigL";
		File fEigL= new File(eigDictL);
		
		String eigDictR=_opt.serializeRun+"EigR";
		File fEigR= new File(eigDictR);
		
			
		try{
			ObjectOutput ccaEig=new ObjectOutputStream(new FileOutputStream(fEig));
			ObjectOutput ccaEigL=new ObjectOutputStream(new FileOutputStream(fEigL));
			ObjectOutput ccaEigR=new ObjectOutputStream(new FileOutputStream(fEigR));
			ObjectOutput ccaContext=new ObjectOutputStream(new FileOutputStream(fContext));
			
		if(_opt.typeofDecomp.equals("TwoStepLRvsW") || _opt.typeofDecomp.equals("LRMVLVariant2") ){
			
				ccaEig.writeObject(phiL);
				ccaEig.flush();
				ccaEig.close();
			
				ccaEigL.writeObject(phiL_1stage);
				ccaEigL.flush();
				ccaEigL.close();
			
				ccaEigR.writeObject(phiR_1stage);
				ccaEigR.flush();
				ccaEigR.close();
	
		
		}else{
			ccaEig.writeObject(phiL);
			ccaEig.flush();
			ccaEig.close();
		
			ccaContext.writeObject(phiR);
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
