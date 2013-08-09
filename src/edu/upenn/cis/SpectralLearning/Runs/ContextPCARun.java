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
import edu.upenn.cis.SpectralLearning.MathUtils.SVDTemplates;
import edu.upenn.cis.SpectralLearning.SpectralRepresentations.ContextPCARepresentation;

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
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream("Output_Files/ContextPCAStats"),"UTF8"));
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
		_cpcaR2.computeTrainLRMatrices();
		SparseDoubleMatrix2D W=_cpcaR2.getWnMatrix();
		SparseDoubleMatrix2D WT=_cpcaR2.getWnTMatrix();
		SparseDoubleMatrix2D L=_cpcaR2.getLnMatrix();
		SparseDoubleMatrix2D LT=_cpcaR2.getLnTMatrix();
		SparseDoubleMatrix2D R=_cpcaR2.getRnMatrix();
		SparseDoubleMatrix2D RT=_cpcaR2.getRnTMatrix();
		
		SparseDoubleMatrix2D wtl =new SparseDoubleMatrix2D(WT.rows(),L.columns());
		SparseDoubleMatrix2D wtlr =new SparseDoubleMatrix2D(WT.rows(),(L.columns()+R.columns()));
		SparseDoubleMatrix2D wtr =new SparseDoubleMatrix2D(WT.rows(),R.columns());
		
		SparseDoubleMatrix2D wtl1 =new SparseDoubleMatrix2D(WT.rows(),L.columns());
		SparseDoubleMatrix2D wtlr1 =new SparseDoubleMatrix2D(WT.rows(),(L.columns()+R.columns()));
		SparseDoubleMatrix2D wtr1 =new SparseDoubleMatrix2D(WT.rows(),R.columns());
		
		SparseDoubleMatrix2D ltw =new SparseDoubleMatrix2D(LT.rows(),W.columns());
		SparseDoubleMatrix2D rtw =new SparseDoubleMatrix2D(RT.rows(),W.columns());
		
		SparseDoubleMatrix2D wtw =new SparseDoubleMatrix2D(WT.rows(),W.columns());
		SparseDoubleMatrix2D lrtw =new SparseDoubleMatrix2D((LT.rows()+RT.rows()),W.columns());
		
		SVDTemplates svdTC;
		
		
		WT.zMult(L, wtl1);
		WT.zMult(R, wtr1);
		LT.zMult(W, ltw);
		RT.zMult(W, rtw);
		
		WT.zMult(W, wtw);
		
		WT.zMult(_cpcaR2.concatenateLR(L, R), wtlr1);
		_cpcaR2.concatenateLRT(LT, RT).zMult(W, lrtw);
		
		
		if(_opt.normalizePCA){
			wtw.zMult(wtl1, wtl);
			wtw.zMult(wtr1, wtr);
			wtw.zMult(wtlr1, wtlr);
		}
		else{
			wtl=wtl1;
			wtr=wtr1;
			wtlr=wtlr1;
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
