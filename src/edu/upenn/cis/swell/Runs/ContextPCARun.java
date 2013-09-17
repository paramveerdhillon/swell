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

import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import Jama.Matrix;
import edu.upenn.cis.swell.IO.Options;
import edu.upenn.cis.swell.MathUtils.MatrixFormatConversion;
import edu.upenn.cis.swell.MathUtils.SVDTemplates;
import edu.upenn.cis.swell.SpectralRepresentations.ContextPCARepresentation;

public class ContextPCARun implements Serializable {

	private Options _opt;
	private ContextPCARepresentation _cpcaR; 
	private Matrix dictMatrixWTL,dictMatrixWTR,dictMatrixLTW,dictMatrixRTW,dictMatrixWTLR,dictMatrixLRTW;
	private int dim2=0;
	static final long serialVersionUID = 42L;
	double[] s=null; 
	public ContextPCARun(Options opt, ContextPCARepresentation cpcaR){
		_opt=opt;
		_cpcaR=cpcaR;
		dim2=2*_opt.contextSizeOneSide*(_opt.vocabSize+1);
		
		System.out.println("+++Entering ContextPCA Compute+++");
		computeCPCA(_cpcaR);
			
		writeStats();
	}


	private void writeStats() {
		BufferedWriter writer=null;
		
		double sSum=0;
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.outputDir+"ContextPCAStats"),"UTF8"));
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


	private void computeCPCA(ContextPCARepresentation _cpcaR2) {
		_cpcaR2.computeContextLRMatrices();
		
		SparseDoubleMatrix2D wtl=null,wtr=null,rtw=null,ltw=null;
		FlexCompRowMatrix wtl1=null,wtr1=null,wtlr1=null;
		
		SparseDoubleMatrix2D wtlr=null;
		SparseDoubleMatrix2D lrtw=null;
		SparseDoubleMatrix2D wtw =new SparseDoubleMatrix2D(_opt.vocabSize+1,_opt.vocabSize+1);
		
		
		SVDTemplates svdTC;
		svdTC=new SVDTemplates(_opt,dim2);
		
		wtw=MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR.getWTWMatrix());
		
		if(_opt.typeofDecomp.equals("WvsL")){
			 wtl =new SparseDoubleMatrix2D(_opt.vocabSize+1,dim2/2);
			  wtl1 =new FlexCompRowMatrix(_opt.vocabSize+1,dim2/2);
			 ltw =new SparseDoubleMatrix2D(dim2/2,_opt.vocabSize+1);
			
			 
			wtl1=_cpcaR.getWTLMatrix();
			ltw=MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR.getLTWMatrix());
			if(_opt.normalizePCA){
				//Scaling rows by the word counts.
				for (MatrixEntry e : wtl1){
					wtl.set(e.row(), e.column(), e.get()/wtw.get(e.row(),e.row()));
				}
				
			}
			else{
				wtl=MatrixFormatConversion.createSparseMatrixCOLT(wtl1);
			}
		}
		
		if(_opt.typeofDecomp.equals("WvsR")){
			 wtr =new SparseDoubleMatrix2D(_opt.vocabSize+1,dim2/2);
			 wtr1 =new FlexCompRowMatrix(_opt.vocabSize+1,dim2/2);
			 rtw =new SparseDoubleMatrix2D(dim2/2,_opt.vocabSize+1);
			
			
			wtr1=_cpcaR.getWTRMatrix();
			rtw=MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR.getRTWMatrix());
			if(_opt.normalizePCA){
				for (MatrixEntry e : wtr1){
					wtr.set(e.row(), e.column(), e.get()/wtw.get(e.row(),e.row()));
				}
				
			}
			else{
				wtr=MatrixFormatConversion.createSparseMatrixCOLT(wtr1);
			}
			
		}
		
		if(_opt.typeofDecomp.equals("WvsLR")){
			
			 lrtw =new SparseDoubleMatrix2D(dim2,_opt.vocabSize+1);
			 wtlr1 =new FlexCompRowMatrix(_opt.vocabSize+1,dim2);
			 wtlr =new SparseDoubleMatrix2D(_opt.vocabSize+1,dim2);
			
			
			wtlr1=_cpcaR.getWTLRMatrix();
			lrtw=MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR.getLRTWMatrix());
			
			wtw=MatrixFormatConversion.createSparseMatrixCOLT(_cpcaR.getWTWMatrix());
			if(_opt.normalizePCA){
				for (MatrixEntry e : wtlr1){
					wtlr.set(e.row(), e.column(), e.get()/wtw.get(e.row(),e.row()));
					
							}
			}
			else{
				wtlr=MatrixFormatConversion.createSparseMatrixCOLT(wtlr1);
			}
			
		}
		
		svdTC=new SVDTemplates(_opt,dim2);

			
		System.out.println("+++Generated the Context Matrix+++");
		
		if(_opt.typeofDecomp.equals("WvsL")){
			dictMatrixWTL=svdTC.computeSVD_Tropp(wtl, _cpcaR2.getOmegaMatrix(wtl.columns()),wtl.columns());
			s=svdTC.getSingularVals();
			dictMatrixLTW=svdTC.computeSVD_Tropp(ltw, _cpcaR2.getOmegaMatrix(ltw.columns()),ltw.columns());
		}
		
		if(_opt.typeofDecomp.equals("WvsR")){
			dictMatrixWTR=svdTC.computeSVD_Tropp(wtr, _cpcaR2.getOmegaMatrix(wtr.columns()),wtr.columns());
			s=svdTC.getSingularVals();
			dictMatrixRTW=svdTC.computeSVD_Tropp(rtw, _cpcaR2.getOmegaMatrix(rtw.columns()),rtw.columns());
		}
		
		if(_opt.typeofDecomp.equals("WvsLR")){
			dictMatrixWTLR=svdTC.computeSVD_Tropp(wtlr, _cpcaR2.getOmegaMatrix(wtlr.columns()),wtlr.columns());
			s=svdTC.getSingularVals();
			dictMatrixLRTW=svdTC.computeSVD_Tropp(lrtw, _cpcaR2.getOmegaMatrix(lrtw.columns()),lrtw.columns());
		}
		
		
		
	}
	
	/*
	public Matrix getEigenDict(){
		return dictMatrix;
	}
	*/
	
	public void serializeContextPCARun() {
		
		
		String contextDict=_opt.serializeRun+"Context";
		File fContext= new File(contextDict);
		
		String eigDict=_opt.serializeRun+"Eig";
		File fEig= new File(eigDict);
			
		try{
			ObjectOutput cpcaEig=new ObjectOutputStream(new FileOutputStream(fEig));
			ObjectOutput cpcaContext=new ObjectOutputStream(new FileOutputStream(fContext));
			
			if(_opt.typeofDecomp.equals("WvsL")){
			
				cpcaEig.writeObject(dictMatrixWTL);
				cpcaEig.flush();
				cpcaEig.close();
			
				cpcaContext.writeObject(dictMatrixLTW);
				cpcaContext.flush();
				cpcaContext.close();
			}
			if(_opt.typeofDecomp.equals("WvsR")){
				
				cpcaEig.writeObject(dictMatrixWTR);
				cpcaEig.flush();
				cpcaEig.close();
			
				cpcaContext.writeObject(dictMatrixRTW);
				cpcaContext.flush();
				cpcaContext.close();
			}
			if(_opt.typeofDecomp.equals("WvsLR")){
				
				cpcaEig.writeObject(dictMatrixWTLR);
				cpcaEig.flush();
				cpcaEig.close();
			
				cpcaContext.writeObject(dictMatrixLRTW);
				cpcaContext.flush();
				cpcaContext.close();
			}
			
			
			System.out.println("=======Serialized the Context PCA Run=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
		
	}
}
