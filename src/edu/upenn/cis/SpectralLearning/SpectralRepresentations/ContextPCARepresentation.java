package edu.upenn.cis.SpectralLearning.SpectralRepresentations;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import Jama.Matrix;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import edu.upenn.cis.SpectralLearning.IO.Options;
import edu.upenn.cis.SpectralLearning.IO.ReadDataFile;
import edu.upenn.cis.SpectralLearning.MathUtils.MatrixFormatConversion;

public class ContextPCARepresentation extends SpectralRepresentation implements Serializable {

	private int _vocab_size;
	private int _contextSize;
	//private Corpus _corpus;
	ReadDataFile _rin;
	SparseDoubleMatrix2D CMatrix_vTimes2hv,LMatrix_hvTimesv,RMatrix_hvTimesv,
	CMatrix_vTimesv,LMatrix_vTimesv,RMatrix_vTimesv,LMatrix_nTimeshv,RMatrix_nTimeshv,
	LMatrix_nTimesv,RMatrix_nTimesv,WMatrix_nTimesv,LTMatrix_nTimeshv,RTMatrix_nTimeshv,
	LTMatrix_nTimesv,RTMatrix_nTimesv,WTMatrix_nTimesv,WMatrix_vTimesv, CTMatrix_vTimes2hv,LTMatrix_hvTimesv,RTMatrix_hvTimesv,
	CTMatrix_vTimesv,LTMatrix_vTimesv,RTMatrix_vTimesv;
	static final long serialVersionUID = 42L;
	long _numTok;
	ArrayList<ArrayList<Integer>> _allDocs;
	
	public ContextPCARepresentation(Options opt, long numTok, ReadDataFile rin,ArrayList<ArrayList<Integer>> all_Docs) {
		super(opt, numTok);
		_vocab_size=super._opt.vocabSize;
		_rin=rin;
		_contextSize=_opt.contextSizeOneSide;
		_allDocs=all_Docs;	
		_numTok=numTok;
	}
	/*
	public void computeLRContextMatrices(){
		int idx_tok,tok;
		HashMap<Double,Double> hMCounts=new HashMap<Double,Double>();
		
		WMatrix_vTimesv=new SparseDoubleMatrix2D((_vocab_size+1),_vocab_size+1);
		if (_opt.bagofWordsSVD){
			CMatrix_vTimesv=new SparseDoubleMatrix2D(_vocab_size+1,(_vocab_size+1));
			LMatrix_vTimesv=new SparseDoubleMatrix2D((_vocab_size+1),_vocab_size+1);
			RMatrix_vTimesv=new SparseDoubleMatrix2D((_vocab_size+1),_vocab_size+1);
			CTMatrix_vTimesv=new SparseDoubleMatrix2D(_vocab_size+1,(_vocab_size+1));
			LTMatrix_vTimesv=new SparseDoubleMatrix2D((_vocab_size+1),_vocab_size+1);
			RTMatrix_vTimesv=new SparseDoubleMatrix2D((_vocab_size+1),_vocab_size+1);
			
			
		}
		else{
			CMatrix_vTimes2hv=new SparseDoubleMatrix2D(_vocab_size+1,2*_contextSize*(_vocab_size+1));
			LMatrix_hvTimesv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_vocab_size+1);
			RMatrix_hvTimesv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_vocab_size+1);
			
			CTMatrix_vTimes2hv=new SparseDoubleMatrix2D(2*_contextSize*(_vocab_size+1),_vocab_size+1);
			LTMatrix_hvTimesv=new SparseDoubleMatrix2D(_vocab_size+1,_contextSize*(_vocab_size+1));
			RTMatrix_hvTimesv=new SparseDoubleMatrix2D(_vocab_size+1,_contextSize*(_vocab_size+1));
		}
		
		int idx_doc=0;
			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				idx_tok=0;
				while(idx_tok<doc.size()){
					tok=doc.get(idx_tok);
					
					//for(int i=1;i<=super._opt.contextSizeOneSide;i++){
						//if (idx_tok-i>=0){
							//int c1=doc.get(idx_tok-i);
							if (_opt.bagofWordsSVD){
								if(hMCounts.get((double)tok) !=null)
									hMCounts.put((double) tok, 1+hMCounts.get((double)tok));
								else
									hMCounts.put((double)tok, 1.0);
								}
							else{
								if(hMCounts.get((double)tok) !=null)
									hMCounts.put((double) tok, 1+hMCounts.get((double)tok));
								else
									hMCounts.put((double)tok, 1.0);
							}
						}
						
						//int j=doc.size();
						if (idx_tok+i <doc.size()){
							//int c=doc.get(idx_tok+i);
							//int ii=_contextSize*(_vocab_size+1);
							if (_opt.bagofWordsSVD){
								if(hMCounts.get((double)tok) !=null)
									hMCounts.put((double) tok, 1+hMCounts.get((double)tok));
								else
									hMCounts.put((double)tok, 1.0);
							}
							else{
								if(hMCounts.get((double)tok) !=null)
									hMCounts.put((double) tok, 1+hMCounts.get((double)tok));
								else
									hMCounts.put((double)tok, 1.0);							
							}
						}
						
					}
					
					idx_tok++;
				}
			}
			
			
			
			
			///////////////
			idx_doc=0;
			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				idx_tok=0;
				while(idx_tok<doc.size()){
					tok=doc.get(idx_tok);
					double countW=WMatrix_vTimesv.get(tok, tok);
					WMatrix_vTimesv.set(tok, tok,countW+(hMCounts.get((double)tok)));
					
					for(int i=1;i<=super._opt.contextSizeOneSide;i++){
						if (idx_tok-i>=0){
							//int c1=doc.get(idx_tok-i);
							if (_opt.bagofWordsSVD){
								double countC=CMatrix_vTimesv.get(tok, doc.get(idx_tok-i));
								double countL=LMatrix_vTimesv.get(doc.get(idx_tok-i),tok);
								CMatrix_vTimesv.set(tok, doc.get(idx_tok-i), countC+(1/hMCounts.get((double)tok)));
								LMatrix_vTimesv.set(doc.get(idx_tok-i),tok, countL+(1/hMCounts.get((double)tok)));
								
								CTMatrix_vTimesv.set( doc.get(idx_tok-i),tok, countC+(1/hMCounts.get((double)tok)));
								LTMatrix_vTimesv.set(tok,doc.get(idx_tok-i), countL+(1/hMCounts.get((double)tok)));
							}
							else{
								double countC=CMatrix_vTimes2hv.get(tok,(i-1)*(_vocab_size+1)+ doc.get(idx_tok-i));
								double countL=LMatrix_hvTimesv.get((i-1)*(_vocab_size+1)+doc.get(idx_tok-i),tok);
								CMatrix_vTimes2hv.set(tok, (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), countC+(1/hMCounts.get((double)tok)));
								LMatrix_hvTimesv.set((i-1)*(_vocab_size+1)+doc.get(idx_tok-i),tok, countL+(1/hMCounts.get((double)tok)));
								
								CTMatrix_vTimes2hv.set( (i-1)*(_vocab_size+1)+doc.get(idx_tok-i),tok, countC+(1/hMCounts.get((double)tok)));
								LTMatrix_hvTimesv.set(tok,(i-1)*(_vocab_size+1)+doc.get(idx_tok-i), countL+(1/hMCounts.get((double)tok)));
							}
						}
						//int j=doc.size();
						if (idx_tok+i <doc.size()){
							//int c=doc.get(idx_tok+i);
							//int ii=_contextSize*(_vocab_size+1);
							if (_opt.bagofWordsSVD){
								double countC=CMatrix_vTimesv.get(tok, doc.get(idx_tok+i));
								double countR=RMatrix_vTimesv.get(doc.get(idx_tok+i),tok);
								CMatrix_vTimesv.set(tok, doc.get(idx_tok+i), countC+(1/hMCounts.get((double)tok)));
								RMatrix_vTimesv.set(doc.get(idx_tok+i),tok, countR+(1/hMCounts.get((double)tok)));
								
								CTMatrix_vTimesv.set( doc.get(idx_tok+i),tok, countC+(1/hMCounts.get((double)tok)));
								RTMatrix_vTimesv.set(tok,doc.get(idx_tok+i), countR+(1/hMCounts.get((double)tok)));
							}
							else{
								double countC=CMatrix_vTimes2hv.get(tok, _contextSize*(_vocab_size+1)+(i-1)*(_vocab_size+1)+doc.get(idx_tok+i));
								double countR=RMatrix_hvTimesv.get((i-1)*(_vocab_size+1)+doc.get(idx_tok+i),tok);
								CMatrix_vTimes2hv.set(tok, _contextSize*(_vocab_size+1)+(i-1)*(_vocab_size+1)+doc.get(idx_tok+i), countC+(1/hMCounts.get((double)tok)));
								RMatrix_hvTimesv.set((i-1)*(_vocab_size+1)+doc.get(idx_tok+i),tok, countR+(1/hMCounts.get((double)tok)));
								
								CTMatrix_vTimes2hv.set(_contextSize*(_vocab_size+1)+(i-1)*(_vocab_size+1)+doc.get(idx_tok+i),tok, countC+(1/hMCounts.get((double)tok)));
								RTMatrix_hvTimesv.set(tok,(i-1)*(_vocab_size+1)+doc.get(idx_tok+i), countR+(1/hMCounts.get((double)tok)));
							}
						}
					}
					idx_tok++;
				}
			}
			
			
		}
	
*/	
	
	public void computeTrainLRMatrices(){
		
		//if (_opt.bagofWordsSVD){
		//	LMatrix_nTimesv=new SparseDoubleMatrix2D((int) _numTok,(_vocab_size+1));
		//	RMatrix_nTimesv=new SparseDoubleMatrix2D((int) _numTok,(_vocab_size+1));
		 //   LTMatrix_nTimesv = new SparseDoubleMatrix2D((_vocab_size+1),(int) _numTok);
		//	RTMatrix_nTimesv=new SparseDoubleMatrix2D((_vocab_size+1),(int) _numTok);
	//	}
		//else{
			LMatrix_nTimeshv=new SparseDoubleMatrix2D((int) _numTok,_contextSize*(_vocab_size+1));
			RMatrix_nTimeshv=new SparseDoubleMatrix2D((int) _numTok,_contextSize*(_vocab_size+1));
			LTMatrix_nTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),(int) _numTok);
			RTMatrix_nTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),(int) _numTok);
		//}
		WMatrix_nTimesv=new SparseDoubleMatrix2D((int) _numTok,(_vocab_size+1));
		WTMatrix_nTimesv=new SparseDoubleMatrix2D((_vocab_size+1),(int) _numTok);
		
		int idx_doc=0;
		int idx_toksAllDocs=0;
			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				int idx_tok=0;
				while(idx_tok<doc.size()){
					int tok=doc.get(idx_tok);
					WMatrix_nTimesv.set(idx_toksAllDocs, tok, 1);
					WTMatrix_nTimesv.set( tok,idx_toksAllDocs, 1);
					for(int i=1;i<=_contextSize;i++){
						if (idx_tok-i>=0){
							//if (_opt.bagofWordsSVD){
							//	LMatrix_nTimesv.set(idx_toksAllDocs, doc.get(idx_tok-i), 1);
							//	LTMatrix_nTimesv.set(doc.get(idx_tok-i),idx_toksAllDocs, 1);
							//}
							//else{
								LMatrix_nTimeshv.set(idx_toksAllDocs, (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
								LTMatrix_nTimeshv.set((i-1)*(_vocab_size+1)+doc.get(idx_tok-i),idx_toksAllDocs, 1);
							//}
						}
						if (idx_tok+i <doc.size()){
							//if (_opt.bagofWordsSVD){
							//	RMatrix_nTimesv.set(idx_toksAllDocs, doc.get(idx_tok+i), 1);
							//	RTMatrix_nTimesv.set(doc.get(idx_tok+i),idx_toksAllDocs, 1);
							//}
							//else{
								RMatrix_nTimeshv.set(idx_toksAllDocs, (i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
								RTMatrix_nTimeshv.set((i-1)*(_vocab_size+1)+doc.get(idx_tok+i),idx_toksAllDocs, 1);
							//}
						}
					}
					idx_tok++;
					idx_toksAllDocs++;
				}
	}	
		
	}
	

	
/*	
	public SparseDoubleMatrix2D getContextMatrix(){
		if (_opt.bagofWordsSVD)
			return CMatrix_vTimesv;
		else
			return CMatrix_vTimes2hv;
	}
	
public SparseDoubleMatrix2D getWTWMatrix(){
		
		return WMatrix_vTimesv;
	
	}
	
	public SparseDoubleMatrix2D getLMatrix(){
		if (_opt.bagofWordsSVD)
			return LMatrix_vTimesv;
		else
			return LMatrix_hvTimesv;
		
	}
	
	
	public SparseDoubleMatrix2D getRMatrix(){
		if (_opt.bagofWordsSVD)
			return RMatrix_vTimesv;
		else
			return RMatrix_hvTimesv;
	}
	
	public SparseDoubleMatrix2D getLTMatrix(){
		if (_opt.bagofWordsSVD)
			return LTMatrix_vTimesv;
		else
			return LTMatrix_hvTimesv;
		
	}
	
	
	public SparseDoubleMatrix2D getRTMatrix(){
		if (_opt.bagofWordsSVD)
			return RTMatrix_vTimesv;
		else
			return RTMatrix_hvTimesv;
	}
*/	
	
	
	public SparseDoubleMatrix2D getWnMatrix(){
		
		return WMatrix_nTimesv;
	}
	
	public SparseDoubleMatrix2D getLnMatrix(){
		//if (_opt.bagofWordsSVD)
			//return LMatrix_nTimesv;
		//else
			return LMatrix_nTimeshv;
		
	}
	
	public SparseDoubleMatrix2D getRnMatrix(){
		//if (_opt.bagofWordsSVD)
			//return RMatrix_nTimesv;
		
		//else
			return RMatrix_nTimeshv;
	}
	
	
public SparseDoubleMatrix2D getWnTMatrix(){
		
		return WTMatrix_nTimesv;
	}
	
	public SparseDoubleMatrix2D getLnTMatrix(){
		//if (_opt.bagofWordsSVD)
		//	return LTMatrix_nTimesv;
		//else
			return LTMatrix_nTimeshv;
		
	}
	
	public SparseDoubleMatrix2D getRnTMatrix(){
		//if (_opt.bagofWordsSVD)
		//	return RTMatrix_nTimesv;
		
	//	else
			return RTMatrix_nTimeshv;
	}
	
	
	public SparseDoubleMatrix2D concatenateLR(SparseDoubleMatrix2D lProjectionMatrix,
			SparseDoubleMatrix2D rProjectionMatrix) {
		SparseDoubleMatrix2D finalProjection=new SparseDoubleMatrix2D(lProjectionMatrix.rows(),(lProjectionMatrix.columns()+rProjectionMatrix.columns()));
		
		for (int i=0;i<lProjectionMatrix.rows();i++){
			for(int j=0; j<lProjectionMatrix.columns();j++){
				finalProjection.set(i, j, lProjectionMatrix.get(i, j));
				finalProjection.set(i, j+lProjectionMatrix.columns(), rProjectionMatrix.get(i, j));
			}
		}
		return finalProjection;
	}
	
	public SparseDoubleMatrix2D concatenateLRT(SparseDoubleMatrix2D lnTMatrix,
			SparseDoubleMatrix2D rnTMatrix) {
		
		SparseDoubleMatrix2D finalProjection=new SparseDoubleMatrix2D((lnTMatrix.rows()+rnTMatrix.rows()),lnTMatrix.columns());
		for (int i=0;i<lnTMatrix.rows();i++){
			for(int j=0; j<lnTMatrix.columns();j++){
				finalProjection.set(i, j, lnTMatrix.get(i, j));
				finalProjection.set(i+lnTMatrix.rows(), j, rnTMatrix.get(i, j));
			}
		}
		return finalProjection;
	}
	
	public DenseDoubleMatrix2D getOmegaMatrix(){//Refer Tropp's notation
		Random r= new Random();
		DenseDoubleMatrix2D Omega;
		//if (_opt.bagofWordsSVD){
		//	Omega= new DenseDoubleMatrix2D((_vocab_size+1),_num_hidden+20);//Oversampled the rank k
		//	for (int i=0;i<(_vocab_size+1);i++){
		//		for (int j=0;j<_num_hidden+20;j++)
		//			Omega.set(i,j,r.nextGaussian());
		//	}
		
		//}else{
			Omega= new DenseDoubleMatrix2D(2*_contextSize*(_vocab_size+1),_num_hidden+20);//Oversampled the rank k
			for (int i=0;i<2*_contextSize*(_vocab_size+1);i++){
				for (int j=0;j<_num_hidden+20;j++)
					Omega.set(i,j,r.nextGaussian());
			}
		//}
		return Omega;
	}
	
	public DenseDoubleMatrix2D getOmegaMatrix(int rows){//Refer Tropp's notation
		Random r= new Random();
		DenseDoubleMatrix2D Omega;
		
			Omega= new DenseDoubleMatrix2D(rows,_num_hidden+20);//Oversampled the rank k
			for (int i=0;i<(rows);i++){
				for (int j=0;j<_num_hidden+20;j++)
					Omega.set(i,j,r.nextGaussian());
			}
		return Omega;
	}
	
	public DenseDoubleMatrix2D getLROmegaMatrix(){//Refer Tropp's notation
		Random r= new Random();
		DenseDoubleMatrix2D Omega= new DenseDoubleMatrix2D((_vocab_size+1),_num_hidden+20);//Oversampled the rank k
		for (int i=0;i<(_vocab_size+1);i++){
			for (int j=0;j<_num_hidden+20;j++)
				Omega.set(i,j,r.nextGaussian());
		}
		return Omega;
	}

	public void serializeContextPCARepresentation() {
		File f= new File(_opt.serializeRep);
		
		try{
			ObjectOutput cpcaRep=new ObjectOutputStream(new FileOutputStream(f));
			cpcaRep.writeObject(this);
			
			System.out.println("=======Serialized the ContextPCA Representation=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
	}
	public Matrix getContextOblEmbeddings(Matrix eigenFeatDict) {
		Matrix WProjectionMatrix;
		
		WProjectionMatrix=generateWProjections(_allDocs,_rin.getSortedWordList(),eigenFeatDict);
		
		
		return WProjectionMatrix;

	}

	public Matrix generateProjections(Matrix matrixEig, Matrix matrixL,
			Matrix matrixR) {
		
		computeTrainLRMatrices();
		DenseDoubleMatrix2D L=new DenseDoubleMatrix2D((int) _numTok,_num_hidden);
		DenseDoubleMatrix2D R=new DenseDoubleMatrix2D((int) _numTok,_num_hidden);
		DenseDoubleMatrix2D W=new DenseDoubleMatrix2D((int) _numTok,_num_hidden);
		DenseDoubleMatrix2D contextSpecificEmbed=new DenseDoubleMatrix2D((int) _numTok,3*_num_hidden);
		
		//if (_opt.bagofWordsSVD){
			//LMatrix_nTimesv.zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixL), L);
			//RMatrix_nTimesv.zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixR), R);
		//}
		//else{
			LMatrix_nTimeshv.zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixL), L);
			RMatrix_nTimeshv.zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixR), R);
		//}
		WMatrix_nTimesv.zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixEig), W);
		
		contextSpecificEmbed=(DenseDoubleMatrix2D)DoubleFactory2D.dense.appendColumns(L, W);
		
		contextSpecificEmbed=(DenseDoubleMatrix2D)DoubleFactory2D.dense.appendColumns(contextSpecificEmbed, R);
		
		return MatrixFormatConversion.createDenseMatrixJAMA(contextSpecificEmbed);
	}

	

		
}
