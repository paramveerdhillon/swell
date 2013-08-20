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
import cern.jet.math.tdouble.DoublePlusMultFirst;
import edu.upenn.cis.SpectralLearning.IO.Options;
import edu.upenn.cis.SpectralLearning.IO.ReadDataFile;
import edu.upenn.cis.SpectralLearning.MathUtils.MatrixFormatConversion;

public class ContextPCARepresentation extends SpectralRepresentation implements Serializable {

	private int _vocab_size;
	private int _contextSize;
	//private Corpus _corpus;
	ReadDataFile _rin;
	SparseDoubleMatrix2D LMatrix_nTimeshv,RMatrix_nTimeshv,
	LMatrix_nTimesv,RMatrix_nTimesv,WMatrix_nTimesv,LTMatrix_nTimeshv,RTMatrix_nTimeshv,
	LTMatrix_nTimesv,RTMatrix_nTimesv,WTMatrix_nTimesv,WMatrix_vTimesv, LTRMatrix_hvTimeshv,RTLMatrix_hvTimeshv,
	LTLMatrix_hvTimeshv,RTRMatrix_hvTimeshv,WTLMatrix_vTimeshv,LTWMatrix_hvTimesv,WTRMatrix_vTimeshv,RTWMatrix_hvTimesv,
	WTLRMatrix_vTimes2hv,LRTWMatrix_2hvTimesv,LRTLRMatrix_2hvTimes2hv;
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
								LMatrix_nTimeshv.set(idx_toksAllDocs, (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
								LTMatrix_nTimeshv.set((i-1)*(_vocab_size+1)+doc.get(idx_tok-i),idx_toksAllDocs, 1);
							//}
						}
						if (idx_tok+i <doc.size()){
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
	
	
	public void computeContextLRMatrices(){
	
		//We can not have n*hv sparse matrices due to limits on max. matrix sizes so we will have to perform the multiplication here only.
		
		if(_opt.typeofDecomp.equals("2viewLvsR") || _opt.typeofDecomp.equals("TwoStepLRvsW") ){
		
			LTRMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
			RTLMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
			LTLMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
			RTRMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
			
			populateMatricesLvsR(LTRMatrix_hvTimeshv,RTLMatrix_hvTimeshv,LTLMatrix_hvTimeshv,RTRMatrix_hvTimeshv);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsL")|| _opt.typeofDecomp.equals("WvsL") || _opt.typeofDecomp.equals("TwoStepLRvsW")){
			WTLMatrix_vTimeshv=new SparseDoubleMatrix2D((_vocab_size+1),_contextSize*(_vocab_size+1));
			LTWMatrix_hvTimesv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),(_vocab_size+1));
			LTLMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
			WMatrix_vTimesv=new SparseDoubleMatrix2D((_vocab_size+1),_vocab_size+1);
			
			populateMatricesWvsL(WTLMatrix_vTimeshv,LTWMatrix_hvTimesv,LTLMatrix_hvTimeshv,WMatrix_vTimesv);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsR")|| _opt.typeofDecomp.equals("WvsR") || _opt.typeofDecomp.equals("TwoStepLRvsW")){
			WTRMatrix_vTimeshv=new SparseDoubleMatrix2D((_vocab_size+1),_contextSize*(_vocab_size+1));
			RTWMatrix_hvTimesv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),(_vocab_size+1));
			RTRMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
			WMatrix_vTimesv=new SparseDoubleMatrix2D((_vocab_size+1),_vocab_size+1);
			
			populateMatricesWvsR(WTRMatrix_vTimeshv,RTWMatrix_hvTimesv,RTRMatrix_hvTimeshv,WMatrix_vTimesv);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsLR")|| _opt.typeofDecomp.equals("WvsLR")){
			WTLRMatrix_vTimes2hv=new SparseDoubleMatrix2D((_vocab_size+1),2*_contextSize*(_vocab_size+1));
			LRTWMatrix_2hvTimesv=new SparseDoubleMatrix2D(2*_contextSize*(_vocab_size+1),(_vocab_size+1));
			LRTLRMatrix_2hvTimes2hv=new SparseDoubleMatrix2D(2*_contextSize*(_vocab_size+1),2*_contextSize*(_vocab_size+1));
			WMatrix_vTimesv=new SparseDoubleMatrix2D((_vocab_size+1),_vocab_size+1);
			
			populateMatricesLRvsW(WTLRMatrix_vTimes2hv,LRTWMatrix_2hvTimesv,LRTLRMatrix_2hvTimes2hv,WMatrix_vTimesv);
		}
				
	}
	
	
	
	private void populateMatricesLRvsW(
			SparseDoubleMatrix2D WTLRMatrix_vTimes2hv,
			SparseDoubleMatrix2D LRTWMatrix_2hvTimesv,
			SparseDoubleMatrix2D LRTLRMatrix_2hvTimes2hv,
			SparseDoubleMatrix2D WMatrix_vTimesv) {
	
		
		int idx_doc=0;
		int idx_toksAllDocs=0;
		SparseDoubleMatrix2D auxMatrixL_nTimeshv,auxMatrixL_hvTimesn,auxMatrixR_nTimeshv,auxMatrixR_hvTimesn,auxMatrix_2hvTimes2hv,
		auxMatrixW_nTimesv,auxMatrixW_vTimesn,auxMatrix_vTimesv,auxMatrix_hvTimes2hv,auxMatrix_hvTimes2hv_1,auxMatrix_hvTimeshv,
		auxMatrix_vTimeshv,auxMatrix_vTimeshv_1,auxMatrix_hvTimesv,auxMatrix_hvTimesv_1;
		
		LTRMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
		RTLMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
		LTLMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
		RTRMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
		
		
		auxMatrix_2hvTimes2hv=new SparseDoubleMatrix2D(2*_contextSize*(_vocab_size+1),2*_contextSize*(_vocab_size+1));
		auxMatrix_vTimesv=new SparseDoubleMatrix2D((_vocab_size+1),(_vocab_size+1));
		auxMatrix_hvTimes2hv_1=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),2*_contextSize*(_vocab_size+1));
		auxMatrix_hvTimes2hv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),2*_contextSize*(_vocab_size+1));
		
		auxMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
		auxMatrix_vTimeshv=new SparseDoubleMatrix2D((_vocab_size+1),_contextSize*(_vocab_size+1));
		auxMatrix_vTimeshv_1=new SparseDoubleMatrix2D((_vocab_size+1),_contextSize*(_vocab_size+1));
		
		auxMatrix_hvTimesv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),(_vocab_size+1));
		auxMatrix_hvTimesv_1=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),(_vocab_size+1));
		
		
			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				int idx_tok=0;
				
				auxMatrixW_nTimesv=new SparseDoubleMatrix2D(doc.size(),(_vocab_size+1));
				auxMatrixW_vTimesn=new SparseDoubleMatrix2D((_vocab_size+1),doc.size());
				
				auxMatrixL_nTimeshv=new SparseDoubleMatrix2D(doc.size(),_contextSize*(_vocab_size+1));
				auxMatrixL_hvTimesn=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),doc.size());
				
				auxMatrixR_nTimeshv=new SparseDoubleMatrix2D(doc.size(),_contextSize*(_vocab_size+1));
				auxMatrixR_hvTimesn=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),doc.size());
				
				
				while(idx_tok<doc.size()){
					int tok=doc.get(idx_tok);
					auxMatrixW_nTimesv.set(idx_tok, tok, 1);
					auxMatrixW_vTimesn.set( tok,idx_tok, 1);
					
					for(int i=1;i<=_contextSize;i++){
						if (idx_tok-i>=0){
								auxMatrixL_nTimeshv.set(idx_tok, (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
								auxMatrixL_hvTimesn.set( (i-1)*(_vocab_size+1)+doc.get(idx_tok-i),idx_tok, 1);
							
						}
						if (idx_tok+i <doc.size()){
								auxMatrixR_nTimeshv.set(idx_tok, (i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
								auxMatrixR_hvTimesn.set( (i-1)*(_vocab_size+1)+doc.get(idx_tok+i),idx_tok, 1);
						}
					}
					idx_tok++;
					idx_toksAllDocs++;
				}
				
				
				auxMatrixL_hvTimesn.zMult(auxMatrixL_nTimeshv,auxMatrix_hvTimeshv);
				LTLMatrix_hvTimeshv.assign(auxMatrix_hvTimeshv,DoublePlusMultFirst.plusMult(1));
				
				auxMatrixR_hvTimesn.zMult(auxMatrixR_nTimeshv,auxMatrix_hvTimeshv);
				RTRMatrix_hvTimeshv.assign(auxMatrix_hvTimeshv,DoublePlusMultFirst.plusMult(1));
				
				
				auxMatrixL_hvTimesn.zMult(auxMatrixR_nTimeshv,auxMatrix_hvTimeshv);
				LTRMatrix_hvTimeshv.assign(auxMatrix_hvTimeshv,DoublePlusMultFirst.plusMult(1));
				
				auxMatrixR_hvTimesn.zMult(auxMatrixL_nTimeshv,auxMatrix_hvTimeshv);
				RTLMatrix_hvTimeshv.assign(auxMatrix_hvTimeshv,DoublePlusMultFirst.plusMult(1));
				
				
				auxMatrix_hvTimes2hv=concatenateLR(LTLMatrix_hvTimeshv,LTRMatrix_hvTimeshv);
				
				auxMatrix_hvTimes2hv_1=concatenateLR(RTLMatrix_hvTimeshv,RTRMatrix_hvTimeshv);
				
				auxMatrix_2hvTimes2hv=concatenateLRT(auxMatrix_hvTimes2hv,auxMatrix_hvTimes2hv_1);
				
				LRTLRMatrix_2hvTimes2hv.assign(auxMatrix_2hvTimes2hv,DoublePlusMultFirst.plusMult(1));
				
				auxMatrixW_vTimesn.zMult(auxMatrixW_nTimesv,auxMatrix_vTimesv);
				WMatrix_vTimesv.assign(auxMatrix_vTimesv,DoublePlusMultFirst.plusMult(1));
				
	
				auxMatrixW_vTimesn.zMult(auxMatrixL_nTimeshv,auxMatrix_vTimeshv);
				auxMatrixW_vTimesn.zMult(auxMatrixR_nTimeshv,auxMatrix_vTimeshv_1);
							
				WTLRMatrix_vTimes2hv.assign(concatenateLR(auxMatrix_vTimeshv,auxMatrix_vTimeshv_1),DoublePlusMultFirst.plusMult(1));
				
				auxMatrixL_hvTimesn.zMult(auxMatrixW_nTimesv,auxMatrix_hvTimesv);
				auxMatrixR_hvTimesn.zMult(auxMatrixW_nTimesv,auxMatrix_hvTimesv_1);
				
				LRTWMatrix_2hvTimesv.assign(concatenateLRT(auxMatrix_hvTimesv,auxMatrix_hvTimesv_1),DoublePlusMultFirst.plusMult(1));			
						
	}
			
	}
	
	
	
	

	public void populateMatricesLvsR(SparseDoubleMatrix2D LTR,SparseDoubleMatrix2D RTL,SparseDoubleMatrix2D LTL,SparseDoubleMatrix2D RTR ){
		int idx_doc=0;
		int idx_toksAllDocs=0;
		SparseDoubleMatrix2D auxMatrixL_nTimeshv,auxMatrixL_hvTimesn,auxMatrixR_nTimeshv,auxMatrixR_hvTimesn,auxMatrix_hvTimeshv;
		
		auxMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
		
			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				int idx_tok=0;
				
				auxMatrixL_nTimeshv=new SparseDoubleMatrix2D(doc.size(),_contextSize*(_vocab_size+1));
				auxMatrixL_hvTimesn=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),doc.size());
				
				auxMatrixR_nTimeshv=new SparseDoubleMatrix2D(doc.size(),_contextSize*(_vocab_size+1));
				auxMatrixR_hvTimesn=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),doc.size());
				
				
				while(idx_tok<doc.size()){
					int tok=doc.get(idx_tok);
					
					
					for(int i=1;i<=_contextSize;i++){
						if (idx_tok-i>=0){
								auxMatrixL_nTimeshv.set(idx_tok, (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
								auxMatrixL_hvTimesn.set( (i-1)*(_vocab_size+1)+doc.get(idx_tok-i),idx_tok, 1);
							
						}
						if (idx_tok+i <doc.size()){
								auxMatrixR_nTimeshv.set(idx_tok, (i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
								auxMatrixR_hvTimesn.set( (i-1)*(_vocab_size+1)+doc.get(idx_tok+i),idx_tok, 1);
						}
					}
					idx_tok++;
					idx_toksAllDocs++;
				}
				
				
				auxMatrixL_hvTimesn.zMult(auxMatrixL_nTimeshv,auxMatrix_hvTimeshv);
				LTLMatrix_hvTimeshv.assign(auxMatrix_hvTimeshv,DoublePlusMultFirst.plusMult(1));
				
				auxMatrixR_hvTimesn.zMult(auxMatrixR_nTimeshv,auxMatrix_hvTimeshv);
				RTRMatrix_hvTimeshv.assign(auxMatrix_hvTimeshv,DoublePlusMultFirst.plusMult(1));
				
				auxMatrixL_hvTimesn.zMult(auxMatrixR_nTimeshv,auxMatrix_hvTimeshv);
				LTRMatrix_hvTimeshv.assign(auxMatrix_hvTimeshv,DoublePlusMultFirst.plusMult(1));
				
				auxMatrixR_hvTimesn.zMult(auxMatrixL_nTimeshv,auxMatrix_hvTimeshv);
				RTLMatrix_hvTimeshv.assign(auxMatrix_hvTimeshv,DoublePlusMultFirst.plusMult(1));
				
	}

	}
	
	
	public void populateMatricesWvsL(SparseDoubleMatrix2D WTL,SparseDoubleMatrix2D LTW,SparseDoubleMatrix2D LTL,SparseDoubleMatrix2D WTW ){
		int idx_doc=0;
		int idx_toksAllDocs=0;
		SparseDoubleMatrix2D auxMatrixW_nTimesv,auxMatrixW_vTimesn, auxMatrixL_nTimeshv,auxMatrixL_hvTimesn,auxMatrix_hvTimeshv
		,auxMatrix_vTimesv,auxMatrix_hvTimesv,auxMatrix_vTimeshv;
		
		
		auxMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
		auxMatrix_vTimesv=new SparseDoubleMatrix2D((_vocab_size+1),(_vocab_size+1));
		auxMatrix_hvTimesv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),(_vocab_size+1));
		auxMatrix_vTimeshv=new SparseDoubleMatrix2D((_vocab_size+1),_contextSize*(_vocab_size+1));
		
			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				int idx_tok=0;
				
				auxMatrixW_nTimesv=new SparseDoubleMatrix2D(doc.size(),(_vocab_size+1));
				auxMatrixW_vTimesn=new SparseDoubleMatrix2D((_vocab_size+1),doc.size());
				
				auxMatrixL_nTimeshv=new SparseDoubleMatrix2D(doc.size(),_contextSize*(_vocab_size+1));
				auxMatrixL_hvTimesn=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),doc.size());
				
				
				while(idx_tok<doc.size()){
					int tok=doc.get(idx_tok);
					
					auxMatrixW_nTimesv.set(idx_tok, tok, 1);
					auxMatrixW_vTimesn.set( tok,idx_tok, 1);
					for(int i=1;i<=_contextSize;i++){
						if (idx_tok-i>=0){
								auxMatrixL_nTimeshv.set(idx_tok, (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
								auxMatrixL_hvTimesn.set( (i-1)*(_vocab_size+1)+doc.get(idx_tok-i),idx_tok, 1);
							
						}
						
					}
					idx_tok++;
					idx_toksAllDocs++;
				}
				
				auxMatrixW_vTimesn.zMult(auxMatrixW_nTimesv,auxMatrix_vTimesv);
				WMatrix_vTimesv.assign(auxMatrix_vTimesv,DoublePlusMultFirst.plusMult(1));
				
				auxMatrixL_hvTimesn.zMult(auxMatrixL_nTimeshv,auxMatrix_hvTimeshv);
				LTLMatrix_hvTimeshv.assign(auxMatrix_hvTimeshv,DoublePlusMultFirst.plusMult(1));
				
				auxMatrixW_vTimesn.zMult(auxMatrixL_nTimeshv,auxMatrix_vTimeshv);
				WTLMatrix_vTimeshv.assign(auxMatrix_vTimeshv,DoublePlusMultFirst.plusMult(1));
				
				auxMatrixL_hvTimesn.zMult(auxMatrixW_nTimesv,auxMatrix_hvTimesv);
				LTWMatrix_hvTimesv.assign(auxMatrix_hvTimesv,DoublePlusMultFirst.plusMult(1));			
				
	}

	}


	
	public void populateMatricesWvsR(SparseDoubleMatrix2D WTR,SparseDoubleMatrix2D RTW,SparseDoubleMatrix2D RTR,SparseDoubleMatrix2D WTW ){
		int idx_doc=0;
		int idx_toksAllDocs=0;
		SparseDoubleMatrix2D auxMatrixW_nTimesv,auxMatrixW_vTimesn, auxMatrixR_nTimeshv,auxMatrixR_hvTimesn,auxMatrix_hvTimeshv
		,auxMatrix_vTimesv,auxMatrix_hvTimesv,auxMatrix_vTimeshv;
		
		
		auxMatrix_hvTimeshv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
		auxMatrix_vTimesv=new SparseDoubleMatrix2D((_vocab_size+1),(_vocab_size+1));
		auxMatrix_hvTimesv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),(_vocab_size+1));
		auxMatrix_vTimeshv=new SparseDoubleMatrix2D((_vocab_size+1),_contextSize*(_vocab_size+1));
		
			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				int idx_tok=0;
				
				auxMatrixW_nTimesv=new SparseDoubleMatrix2D(doc.size(),(_vocab_size+1));
				auxMatrixW_vTimesn=new SparseDoubleMatrix2D((_vocab_size+1),doc.size());
				
				auxMatrixR_nTimeshv=new SparseDoubleMatrix2D(doc.size(),_contextSize*(_vocab_size+1));
				auxMatrixR_hvTimesn=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),doc.size());
				
				
				while(idx_tok<doc.size()){
					int tok=doc.get(idx_tok);
					
					auxMatrixW_nTimesv.set(idx_tok, tok, 1);
					auxMatrixW_vTimesn.set( tok,idx_tok, 1);
					for(int i=1;i<=_contextSize;i++){
						if (idx_tok+i <doc.size()){
							auxMatrixR_nTimeshv.set(idx_tok, (i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
							auxMatrixR_hvTimesn.set( (i-1)*(_vocab_size+1)+doc.get(idx_tok+i),idx_tok, 1);
					}
						
					}
					idx_tok++;
					idx_toksAllDocs++;
				}
				
				auxMatrixW_vTimesn.zMult(auxMatrixW_nTimesv,auxMatrix_vTimesv);
				WMatrix_vTimesv.assign(auxMatrix_vTimesv,DoublePlusMultFirst.plusMult(1));
				
				auxMatrixR_hvTimesn.zMult(auxMatrixR_nTimeshv,auxMatrix_hvTimeshv);
				RTRMatrix_hvTimeshv.assign(auxMatrix_hvTimeshv,DoublePlusMultFirst.plusMult(1));
				
				auxMatrixW_vTimesn.zMult(auxMatrixR_nTimeshv,auxMatrix_vTimeshv);
				WTRMatrix_vTimeshv.assign(auxMatrix_vTimeshv,DoublePlusMultFirst.plusMult(1));
				
				auxMatrixR_hvTimesn.zMult(auxMatrixW_nTimesv,auxMatrix_hvTimesv);
				RTWMatrix_hvTimesv.assign(auxMatrix_hvTimesv,DoublePlusMultFirst.plusMult(1));			
				
	}

	}
	
	

	public SparseDoubleMatrix2D getLTRMatrix(){
		return LTRMatrix_hvTimeshv;
	}
	
	public SparseDoubleMatrix2D getRTLMatrix(){
		return RTLMatrix_hvTimeshv;
	}
	
	public SparseDoubleMatrix2D getLTLMatrix(){
		return LTLMatrix_hvTimeshv;
	}
	
	public SparseDoubleMatrix2D getRTRMatrix(){
		return RTRMatrix_hvTimeshv;
	}
	
	public SparseDoubleMatrix2D getWTLMatrix(){
		return WTLMatrix_vTimeshv;
	}
	
	public SparseDoubleMatrix2D getLTWMatrix(){
		return LTWMatrix_hvTimesv;
	}
	
	public SparseDoubleMatrix2D getWTRMatrix(){
		return WTRMatrix_vTimeshv;
	}
	
	public SparseDoubleMatrix2D getRTWMatrix(){
		return RTWMatrix_hvTimesv;
	}
	
	public SparseDoubleMatrix2D getWTLRMatrix(){
		return WTLRMatrix_vTimes2hv;
	}
	
	public SparseDoubleMatrix2D getLRTWMatrix(){
		return LRTWMatrix_2hvTimesv;
	}
	
	public SparseDoubleMatrix2D getLRTLRMatrix(){
		return LRTLRMatrix_2hvTimes2hv;
	}
	
	public SparseDoubleMatrix2D getWTWMatrix(){
		
		return WMatrix_vTimesv;
	
	}
	
	
	public SparseDoubleMatrix2D getWnMatrix(){
		
		return WMatrix_nTimesv;
	}
	
	public SparseDoubleMatrix2D getLnMatrix(){
			return LMatrix_nTimeshv;
		
	}
	
	public SparseDoubleMatrix2D getRnMatrix(){
			return RMatrix_nTimeshv;
	}
	
	
public SparseDoubleMatrix2D getWnTMatrix(){
		
		return WTMatrix_nTimesv;
	}
	
	public SparseDoubleMatrix2D getLnTMatrix(){
			return LTMatrix_nTimeshv;
		
	}
	
	public SparseDoubleMatrix2D getRnTMatrix(){
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
			Omega= new DenseDoubleMatrix2D(2*_contextSize*(_vocab_size+1),_num_hidden+20);//Oversampled the rank k
			for (int i=0;i<2*_contextSize*(_vocab_size+1);i++){
				for (int j=0;j<_num_hidden+20;j++)
					Omega.set(i,j,r.nextGaussian());
			}
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
		
		WProjectionMatrix=generateWProjections(_rin.getSortedWordList(),eigenFeatDict);
		
		
		return WProjectionMatrix;

	}
	
	protected Matrix generateWProjections(ArrayList<Integer> sortedWordList, Matrix eigenFeatDict) {
		ArrayList<Integer> doc;

			int count=0;
			
		double[][] wProjection=new double[(int)_num_tokens][_num_hidden];
		int idxDoc=0;
		
		while (idxDoc<_allDocs.size()){
			
				doc=_allDocs.get(idxDoc++);
				int idxTok=0;
				while(idxTok<doc.size()){
					for (int j=0;j<_num_hidden;j++){
						wProjection[count][j]=eigenFeatDict.get(doc.get(idxTok), j);
					}
					count++;
					idxTok++;
				}
			}
		return new Matrix(wProjection);
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
