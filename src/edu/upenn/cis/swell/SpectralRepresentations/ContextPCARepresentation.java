package edu.upenn.cis.swell.SpectralRepresentations;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import Jama.Matrix;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import edu.upenn.cis.swell.IO.Options;
import edu.upenn.cis.swell.IO.ReadDataFile;
import edu.upenn.cis.swell.MathUtils.MatrixFormatConversion;

public class ContextPCARepresentation extends SpectralRepresentation implements Serializable {

	private int _vocab_size;
	private int _contextSize,k_dim;
	ReadDataFile _rin;
	FlexCompRowMatrix WMatrix_vTimesv, LTRMatrix_hvTimeshv,RTLMatrix_hvTimeshv,
	LTLMatrix_hvTimeshv,RTRMatrix_hvTimeshv,WTLMatrix_vTimeshv,LTWMatrix_hvTimesv,WTRMatrix_vTimeshv,RTWMatrix_hvTimesv,
	 WTLRMatrix_vTimes2hv,LRTWMatrix_2hvTimesv,LRTLRMatrix_2hvTimes2hv;
	
	DenseDoubleMatrix2D LTRMatrix_hkTimeshk,RTLMatrix_hkTimeshk,
	LTLMatrix_hkTimeshk,RTRMatrix_hkTimeshk,WTLMatrix_vTimeshk,LTWMatrix_hkTimesv,WTRMatrix_vTimeshk,RTWMatrix_hkTimesv,
	 WTLRMatrix_vTimes2hk,LRTWMatrix_2hkTimesv,LRTLRMatrix_2hkTimes2hk;
	
	Matrix kdimEigDict=null;
	
	DenseDoubleMatrix2D LMatrix_nTimeshk,RMatrix_nTimeshk,
	LTMatrix_nTimeshk,RTMatrix_nTimeshk,LRMatrix_nTimes2hk,LRTMatrix_2hkTimesn;
	
	
	
	SparseDoubleMatrix2D LMatrix_nTimeshv,RMatrix_nTimeshv,
	LMatrix_nTimesv,RMatrix_nTimesv,WMatrix_nTimesv,LTMatrix_nTimeshv,RTMatrix_nTimeshv,
	LTMatrix_nTimesv,RTMatrix_nTimesv,WTMatrix_nTimesv;
	
	static final long serialVersionUID = 42L;
	int _numTok;
	ArrayList<ArrayList<Integer>> _allDocs;
	HashMap<Integer, Integer> _wordMap= null;
	
	public ContextPCARepresentation(Options opt, long numTok, ReadDataFile rin,ArrayList<ArrayList<Integer>> all_Docs) {
		super(opt, numTok);
		_vocab_size=super._opt.vocabSize;
		_rin=rin;
		_contextSize=_opt.contextSizeOneSide;
		_allDocs=all_Docs;	
		_numTok=(int)numTok;
	}
		
	public ContextPCARepresentation(Options opt, long numTokens,
			ReadDataFile rin, ArrayList<ArrayList<Integer>> all_Docs,
			Matrix kDimCCADict, HashMap<Integer, Integer> wMap) {
		super(opt, numTokens);
		_vocab_size=super._opt.vocabSize;
		_rin=rin;
		_contextSize=_opt.contextSizeOneSide;
		_allDocs=all_Docs;	
		_numTok=(int)numTokens;
		_wordMap= new HashMap<Integer, Integer>();
		
		_wordMap=wMap;
		kdimEigDict=kDimCCADict;
		k_dim=kdimEigDict.getColumnDimension();
		
	}
	

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
		
		if( _opt.typeofDecomp.equals("TwoStepLRvsW") ){
		
			LTRMatrix_hvTimeshv=new FlexCompRowMatrix(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
			RTLMatrix_hvTimeshv=new FlexCompRowMatrix(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
			LTLMatrix_hvTimeshv=new FlexCompRowMatrix(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
			RTRMatrix_hvTimeshv=new FlexCompRowMatrix(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
			
			WTLMatrix_vTimeshv=new FlexCompRowMatrix((_vocab_size+1),_contextSize*(_vocab_size+1));
			LTWMatrix_hvTimesv=new FlexCompRowMatrix(_contextSize*(_vocab_size+1),(_vocab_size+1));
			
			WMatrix_vTimesv=new FlexCompRowMatrix((_vocab_size+1),_vocab_size+1);
			
			WTRMatrix_vTimeshv=new FlexCompRowMatrix((_vocab_size+1),_contextSize*(_vocab_size+1));
			RTWMatrix_hvTimesv=new FlexCompRowMatrix(_contextSize*(_vocab_size+1),(_vocab_size+1));
			
			populateMatricesTwoStepLRvsW(LTRMatrix_hvTimeshv,RTLMatrix_hvTimeshv,LTLMatrix_hvTimeshv,RTRMatrix_hvTimeshv,WTLMatrix_vTimeshv,
					LTWMatrix_hvTimesv,WMatrix_vTimesv,WTRMatrix_vTimeshv,RTWMatrix_hvTimesv);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsL")|| _opt.typeofDecomp.equals("WvsL")){
			WTLMatrix_vTimeshv=new FlexCompRowMatrix((_vocab_size+1),_contextSize*(_vocab_size+1));
			LTWMatrix_hvTimesv=new FlexCompRowMatrix(_contextSize*(_vocab_size+1),(_vocab_size+1));
			LTLMatrix_hvTimeshv=new FlexCompRowMatrix(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
			WMatrix_vTimesv=new FlexCompRowMatrix((_vocab_size+1),_vocab_size+1);
			
			populateMatricesWvsL(WTLMatrix_vTimeshv,LTWMatrix_hvTimesv,LTLMatrix_hvTimeshv,WMatrix_vTimesv);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsR")|| _opt.typeofDecomp.equals("WvsR") ){
			WTRMatrix_vTimeshv=new FlexCompRowMatrix((_vocab_size+1),_contextSize*(_vocab_size+1));
			RTWMatrix_hvTimesv=new FlexCompRowMatrix(_contextSize*(_vocab_size+1),(_vocab_size+1));
			RTRMatrix_hvTimeshv=new FlexCompRowMatrix(_contextSize*(_vocab_size+1),_contextSize*(_vocab_size+1));
			WMatrix_vTimesv=new FlexCompRowMatrix((_vocab_size+1),_vocab_size+1);
			
			populateMatricesWvsR(WTRMatrix_vTimeshv,RTWMatrix_hvTimesv,RTRMatrix_hvTimeshv,WMatrix_vTimesv);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsLR")|| _opt.typeofDecomp.equals("WvsLR")){
			
			WTLRMatrix_vTimes2hv =new FlexCompRowMatrix((_vocab_size+1),2*_contextSize*(_vocab_size+1));
			
			LRTWMatrix_2hvTimesv=new FlexCompRowMatrix(2*_contextSize*(_vocab_size+1),(_vocab_size+1));
			LRTLRMatrix_2hvTimes2hv=new FlexCompRowMatrix(2*_contextSize*(_vocab_size+1),2*_contextSize*(_vocab_size+1));
			WMatrix_vTimesv=new FlexCompRowMatrix((_vocab_size+1),(_vocab_size+1));
			
			
			System.out.println("+++Initialized the required matrices+++");
			
			
			populateMatricesLRvsW(WTLRMatrix_vTimes2hv,LRTWMatrix_2hvTimesv,LRTLRMatrix_2hvTimes2hv,WMatrix_vTimesv);
			
		}
				
	}
	
	
	
	private void populateMatricesTwoStepLRvsW(
			FlexCompRowMatrix LTR,
			FlexCompRowMatrix RTL,
			FlexCompRowMatrix LTL,
			FlexCompRowMatrix RTR,
			FlexCompRowMatrix WTL,
			FlexCompRowMatrix LTW,
			FlexCompRowMatrix WTW,
			FlexCompRowMatrix WTR,
			FlexCompRowMatrix RTW) {
		
		int idx_doc=0;
		System.out.println("+++Entering loop over the documents+++");
		while (idx_doc<_allDocs.size()){
			ArrayList<Integer> doc=_allDocs.get(idx_doc++);
			int idx_tok=0;
			
			while(idx_tok<doc.size()){
				int tok=doc.get(idx_tok);
				WTW.add(tok, tok, 1);
				
				for(int i=1;i<=_contextSize;i++){
					if (idx_tok-i>=0){
						
						
							WTL.add(tok, (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
							LTW.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i),tok, 1);
						
							
							//LTL
							LTL.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i), (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
							
							//LTR
							if (idx_tok+i <doc.size()){
								LTR.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i), (i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
							}
					}
					if (idx_tok+i <doc.size()){
						
							RTW.add((i-1)*(_vocab_size+1)+doc.get(idx_tok+i), tok, 1);
							WTR.add(tok,(i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
						
							
							//RTR
							RTR.add((i-1)*(_vocab_size+1)+doc.get(idx_tok+i), (i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
							
							//RTL
							if (idx_tok-i>=0){
								RTL.add((i-1)*(_vocab_size+1)+doc.get(idx_tok+i), (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
							}
					}
				}
				idx_tok++;
			}
			if(idx_doc%1000==0)
				System.out.println("++Doc Num: "+idx_doc+" Processed");
		}
	
		////////////
		System.out.println("+++Iterated over the documents+++");

		
	}

	private void populateMatricesLRvsW(
			FlexCompRowMatrix WTLRMatrix_vTimes2hv,
			FlexCompRowMatrix LRTWMatrix_2hvTimesv,
			FlexCompRowMatrix LRTLRMatrix_2hvTimes2hv,
			FlexCompRowMatrix WMatrix_vTimesv) {
	
		
		int idx_doc=0;
		
		
		System.out.println("+++Entering loop over the documents+++");
			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				int idx_tok=0;
				
				while(idx_tok<doc.size()){
					int tok=doc.get(idx_tok);
					WMatrix_vTimesv.add(tok, tok, 1);
					
					
					for(int i=1;i<=_contextSize;i++){
						if (idx_tok-i>=0){
								WTLRMatrix_vTimes2hv.add(tok, (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
								
								LRTWMatrix_2hvTimesv.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i),tok, 1);
								
								//LTL
								LRTLRMatrix_2hvTimes2hv.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i), (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
								
								//LTR
								if (idx_tok+i <doc.size()){
									LRTLRMatrix_2hvTimes2hv.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i), ((_opt.contextSizeOneSide)*(_vocab_size+1))+(i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
								}
						}
						if (idx_tok+i <doc.size()){
							
								WTLRMatrix_vTimes2hv.add(tok, ((_opt.contextSizeOneSide)*(_vocab_size+1))+(i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
								
								LRTWMatrix_2hvTimesv.add(((_opt.contextSizeOneSide)*(_vocab_size+1))+(i-1)*(_vocab_size+1)+doc.get(idx_tok+i),tok, 1);
								
								//RTR
								LRTLRMatrix_2hvTimes2hv.add(((_opt.contextSizeOneSide)*(_vocab_size+1))+(i-1)*(_vocab_size+1)+doc.get(idx_tok+i), ((_opt.contextSizeOneSide)*(_vocab_size+1))+(i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
								
								//RTL
								if (idx_tok-i>=0){
									LRTLRMatrix_2hvTimes2hv.add(((_opt.contextSizeOneSide)*(_vocab_size+1))+(i-1)*(_vocab_size+1)+doc.get(idx_tok+i), (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
								}
						}
					}
					idx_tok++;
				}
				
					if(idx_doc%1000==0)
						System.out.println("++Doc Num: "+idx_doc+" Processed");
			}
		
			
			////////////
			System.out.println("+++Iterated over the documents+++");
			
			
				}
	

		public void populateMatricesLvsR(FlexCompRowMatrix LTR,FlexCompRowMatrix RTL,FlexCompRowMatrix LTL,FlexCompRowMatrix RTR ){
			int idx_doc=0;
			
			
			System.out.println("+++Entering loop over the documents+++");
				while (idx_doc<_allDocs.size()){
					ArrayList<Integer> doc=_allDocs.get(idx_doc++);
					int idx_tok=0;
					
					while(idx_tok<doc.size()){
						int tok=doc.get(idx_tok);
						
						
						for(int i=1;i<=_contextSize;i++){
							if (idx_tok-i>=0){
									
									//LTL
									LTL.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i), (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
									
									//LTR
									if (idx_tok+i <doc.size()){
										LTR.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i), (i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
									}
							}
							if (idx_tok+i <doc.size()){
									
									//RTR
									RTR.add((i-1)*(_vocab_size+1)+doc.get(idx_tok+i), (i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
									
									//RTL
									if (idx_tok-i>=0){
										RTL.add((i-1)*(_vocab_size+1)+doc.get(idx_tok+i), (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
									}
							}
						}
						idx_tok++;
					}
					if(idx_doc%1000==0)
						System.out.println("+Doc Num: "+idx_doc+" Processed");
				}
			
				////////////
				System.out.println("+++Iterated over the documents+++");
				
	}
	
	
	public void populateMatricesWvsL(FlexCompRowMatrix WTL,FlexCompRowMatrix LTW,FlexCompRowMatrix LTL,FlexCompRowMatrix WTW ){
		
		int idx_doc=0;
		System.out.println("+++Entering loop over the documents+++");
		while (idx_doc<_allDocs.size()){
			ArrayList<Integer> doc=_allDocs.get(idx_doc++);
			int idx_tok=0;
			
			while(idx_tok<doc.size()){
				int tok=doc.get(idx_tok);
				WTW.add(tok, tok, 1);
				
				for(int i=1;i<=_contextSize;i++){
					if (idx_tok-i>=0){		
							WTL.add(tok, (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
							LTW.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i),tok, 1);
							
							LTL.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i),(i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
					}
					
				}
				idx_tok++;
			}
			if(idx_doc%1000==0)
				System.out.println("+Doc Num: "+idx_doc+" Processed");
		}
	
		////////////
		System.out.println("+++Iterated over the documents+++");
		

	}


	
	public void populateMatricesWvsR(FlexCompRowMatrix WTR,FlexCompRowMatrix RTW,FlexCompRowMatrix RTR,FlexCompRowMatrix WTW ){
		
		int idx_doc=0;
		
		System.out.println("+++Entering loop over the documents+++");
			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				int idx_tok=0;
				
				while(idx_tok<doc.size()){
					int tok=doc.get(idx_tok);
					WTW.add(tok, tok, 1);
					
					for(int i=1;i<=_contextSize;i++){
						
						if (idx_tok+i <doc.size()){
								
								RTR.add((i-1)*(_vocab_size+1)+doc.get(idx_tok+i), (i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
								RTW.add((i-1)*(_vocab_size+1)+doc.get(idx_tok+i), tok, 1);
								WTR.add(tok,(i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
								
						}
					}
					idx_tok++;
				}
				if(idx_doc%1000==0)
					System.out.println("+Doc Num: "+idx_doc+" Processed");
			}
		
			////////////
			System.out.println("+++Iterated over the documents+++");
	}
	
	
	
	
	public void computeContextLRDenseMatrices(){
		
		
		if( _opt.typeofDecomp.equals("TwoStepLRvsW") ){
		
			//LTRMatrix_hkTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),_contextSize*(k_dim));
			//RTLMatrix_hkTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),_contextSize*(k_dim));
			//LTLMatrix_hkTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),_contextSize*(k_dim));
			//RTRMatrix_hkTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),_contextSize*(k_dim));
			
			WTLMatrix_vTimeshk=new DenseDoubleMatrix2D((_vocab_size+1),_contextSize*(k_dim));
			LTWMatrix_hkTimesv=new DenseDoubleMatrix2D(_contextSize*(k_dim),(_vocab_size+1));
			
			WMatrix_vTimesv=new FlexCompRowMatrix((_vocab_size+1),_vocab_size+1);
			
			WTRMatrix_vTimeshk=new DenseDoubleMatrix2D((_vocab_size+1),_contextSize*(k_dim));
			RTWMatrix_hkTimesv=new DenseDoubleMatrix2D(_contextSize*(k_dim),(_vocab_size+1));
			
			populateMatricesTwoStepLRvsW(WTLMatrix_vTimeshk,LTWMatrix_hkTimesv,WMatrix_vTimesv,WTRMatrix_vTimeshk,RTWMatrix_hkTimesv);
			computeCovMatrices();
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsL")|| _opt.typeofDecomp.equals("WvsL")){
			WTLMatrix_vTimeshk=new DenseDoubleMatrix2D((_vocab_size+1),_contextSize*(k_dim));
			LTWMatrix_hkTimesv=new DenseDoubleMatrix2D(_contextSize*(k_dim),(_vocab_size+1));
			WMatrix_vTimesv=new FlexCompRowMatrix((_vocab_size+1),_vocab_size+1);
			
			populateMatricesWvsL(WTLMatrix_vTimeshk,LTWMatrix_hkTimesv,WMatrix_vTimesv);
			computeCovMatrices();
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsR")|| _opt.typeofDecomp.equals("WvsR") ){
			WTRMatrix_vTimeshk=new DenseDoubleMatrix2D((_vocab_size+1),_contextSize*(k_dim));
			RTWMatrix_hkTimesv=new DenseDoubleMatrix2D(_contextSize*(k_dim),(_vocab_size+1));
			WMatrix_vTimesv=new FlexCompRowMatrix((_vocab_size+1),_vocab_size+1);
			
			populateMatricesWvsR(WTRMatrix_vTimeshk,RTWMatrix_hkTimesv,WMatrix_vTimesv);
			computeCovMatrices();
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsLR")|| _opt.typeofDecomp.equals("WvsLR")){
			
			WTLRMatrix_vTimes2hk =new DenseDoubleMatrix2D((_vocab_size+1),2*_contextSize*(k_dim));
			LRTWMatrix_2hkTimesv=new DenseDoubleMatrix2D(2*_contextSize*(k_dim),(_vocab_size+1));
			WMatrix_vTimesv=new FlexCompRowMatrix((_vocab_size+1),(_vocab_size+1));
			
			
			System.out.println("+++Initialized the required matrices+++");
			
			
			populateMatricesLRvsW(WTLRMatrix_vTimes2hk,LRTWMatrix_2hkTimesv,WMatrix_vTimesv);
			computeCovMatrices();
			
		}
				
	}
	
	private void populateMatricesTwoStepLRvsW(
			DenseDoubleMatrix2D WTL,
			DenseDoubleMatrix2D LTW,
			FlexCompRowMatrix WTW,
			DenseDoubleMatrix2D WTR,
			DenseDoubleMatrix2D RTW) {
		
		int idx_doc=0;
		System.out.println("+++Entering loop over the documents+++");
		while (idx_doc<_allDocs.size()){
			ArrayList<Integer> doc=_allDocs.get(idx_doc++);
			int idx_tok=0;
			
			while(idx_tok<doc.size()){
				int tok=doc.get(idx_tok);
				
				WTW.add(tok, tok, 1);
				
				for(int i=1;i<=_contextSize;i++){
					if (idx_tok-i>=0){
						
						
							for (int j=0;j<k_dim;j++){	
								LTW.setQuick((i-1)*(k_dim)+j, tok, kdimEigDict.get(_wordMap.get(doc.get(idx_tok-i)), j));
								WTL.setQuick(tok,(i-1)*(k_dim)+j, kdimEigDict.get(_wordMap.get(doc.get(idx_tok-i)), j));	
							}
							
							//LTL
							//LTL.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i), (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
							
							//LTR
							//if (idx_tok+i <doc.size()){
							//	LTR.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i), (i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
							//}
					}
					if (idx_tok+i <doc.size()){
						
							
							for (int j=0;j<k_dim;j++){	
								RTW.setQuick((i-1)*(k_dim)+j, tok, kdimEigDict.get(_wordMap.get(doc.get(idx_tok+i)), j));
								WTR.setQuick(tok,(i-1)*(k_dim)+j, kdimEigDict.get(_wordMap.get(doc.get(idx_tok+i)), j));	
							}
										
							
							//RTR
							//RTR.add((i-1)*(_vocab_size+1)+doc.get(idx_tok+i), (i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
							
							//RTL
							//if (idx_tok-i>=0){
							//	RTL.add((i-1)*(_vocab_size+1)+doc.get(idx_tok+i), (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
							//}
					}
				}
				idx_tok++;
			}
			if(idx_doc%1000==0)
				System.out.println("++Doc Num: "+idx_doc+" Processed");
		}
	
		////////////
		System.out.println("+++Iterated over the documents+++");

		
	}
	
	
	private void populateMatricesLRvsW(
			DenseDoubleMatrix2D WTLRMatrix_vTimes2hk,
			DenseDoubleMatrix2D LRTWMatrix_2hkTimesv,
			FlexCompRowMatrix WMatrix_vTimesv) {
	
		
		int idx_doc=0;
		
		
		System.out.println("+++Entering loop over the documents+++");
			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				int idx_tok=0;
				
				while(idx_tok<doc.size()){
					int tok=doc.get(idx_tok);
					WMatrix_vTimesv.add(tok, tok, 1);
					
					
					for(int i=1;i<=_contextSize;i++){
						if (idx_tok-i>=0){
								
								for (int j=0;j<k_dim;j++){	
									LRTWMatrix_2hkTimesv.setQuick((i-1)*(k_dim)+j, tok, kdimEigDict.get(_wordMap.get(doc.get(idx_tok-i)), j));
									WTLRMatrix_vTimes2hk.setQuick(tok,(i-1)*(k_dim)+j, kdimEigDict.get(_wordMap.get(doc.get(idx_tok-i)), j));	
								}
								
								
								//LTL
								//LRTLRMatrix_2hvTimes2hv.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i), (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
								
								//LTR
								//if (idx_tok+i <doc.size()){
								//	LRTLRMatrix_2hvTimes2hv.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i), ((_opt.contextSizeOneSide)*(_vocab_size+1))+(i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
								//}
						}
						if (idx_tok+i <doc.size()){
							
							for (int j=0;j<k_dim;j++){	
								
								LRTWMatrix_2hkTimesv.setQuick(((_opt.contextSizeOneSide)*(k_dim))+(i-1)*(k_dim)+j, tok, kdimEigDict.get(_wordMap.get(doc.get(idx_tok+i)), j));
								WTLRMatrix_vTimes2hk.setQuick(tok,((_opt.contextSizeOneSide)*(k_dim))+(i-1)*(k_dim)+j, kdimEigDict.get(_wordMap.get(doc.get(idx_tok+i)), j));	
							}
								
							
								//RTR
								//LRTLRMatrix_2hvTimes2hv.add(((_opt.contextSizeOneSide)*(_vocab_size+1))+(i-1)*(_vocab_size+1)+doc.get(idx_tok+i), ((_opt.contextSizeOneSide)*(_vocab_size+1))+(i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
								
								//RTL
								//if (idx_tok-i>=0){
								//	LRTLRMatrix_2hvTimes2hv.add(((_opt.contextSizeOneSide)*(_vocab_size+1))+(i-1)*(_vocab_size+1)+doc.get(idx_tok+i), (i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
								//}
						}
					}
					idx_tok++;
				}
				
					if(idx_doc%1000==0)
						System.out.println("++Doc Num: "+idx_doc+" Processed");
			}
		
			
			////////////
			System.out.println("+++Iterated over the documents+++");
			
			
				}

	
	
public void populateMatricesWvsR(DenseDoubleMatrix2D WTR,DenseDoubleMatrix2D RTW,FlexCompRowMatrix WTW ){
		
		int idx_doc=0;
		
		System.out.println("+++Entering loop over the documents+++");
			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				int idx_tok=0;
				
	
				while(idx_tok<doc.size()){
					int tok=doc.get(idx_tok);
					WTW.add(tok, tok, 1);
					
					for(int i=1;i<=_contextSize;i++){
						
						if (idx_tok+i <doc.size()){
							
							for (int j=0;j<k_dim;j++){	
								RTW.setQuick((i-1)*(k_dim)+j, tok, kdimEigDict.get(_wordMap.get(doc.get(idx_tok+i)), j));
								WTR.setQuick(tok,(i-1)*(k_dim)+j, kdimEigDict.get(_wordMap.get(doc.get(idx_tok+i)), j));
								
								
								//RTR.setQuick()
								
								//RTR.setQuick((i-1)*(_vocab_size+1)+doc.get(idx_tok+i), (i-1)*(_vocab_size+1)+doc.get(idx_tok+i), 1);
								
							}
						}
					}
					idx_tok++;
				}
				if(idx_doc%1000==0)
					System.out.println("+Doc Num: "+idx_doc+" Processed");
			}
		
			////////////
			System.out.println("+++Iterated over the documents+++");
	}
	
public void populateMatricesWvsL(DenseDoubleMatrix2D WTL,DenseDoubleMatrix2D LTW,FlexCompRowMatrix WTW ){
	
	int idx_doc=0;
	System.out.println("+++Entering loop over the documents+++");
	while (idx_doc<_allDocs.size()){
		ArrayList<Integer> doc=_allDocs.get(idx_doc++);
		int idx_tok=0;
		
		while(idx_tok<doc.size()){
			int tok=doc.get(idx_tok);
			WTW.add(tok, tok, 1);
			
			for(int i=1;i<=_contextSize;i++){
				if (idx_tok-i>=0){
					
					for (int j=0;j<k_dim;j++){	
						LTW.setQuick((i-1)*(k_dim)+j, tok, kdimEigDict.get(_wordMap.get(doc.get(idx_tok-i)), j));
						WTL.setQuick(tok,(i-1)*(k_dim)+j, kdimEigDict.get(_wordMap.get(doc.get(idx_tok-i)), j));
						
						//LTL.add((i-1)*(_vocab_size+1)+doc.get(idx_tok-i),(i-1)*(_vocab_size+1)+doc.get(idx_tok-i), 1);
						
					}					
				}
				
			}
			idx_tok++;
		}
		if(idx_doc%1000==0)
			System.out.println("+Doc Num: "+idx_doc+" Processed");
	}

	////////////
	System.out.println("+++Iterated over the documents+++");
}

public void computeCovMatrices(){
	
	//This is required as we can not compute LTL, RTR, RTL, LTR, LRTLR from v*hk matrices
	
if(_opt.typeofDecomp.equals("2viewWvsL")|| _opt.typeofDecomp.equals("WvsL")){	
	LMatrix_nTimeshk=new DenseDoubleMatrix2D((int) _numTok,_contextSize*(k_dim));
	LTMatrix_nTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),(int) _numTok);
	LTLMatrix_hkTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),_contextSize*(k_dim));
}

if(_opt.typeofDecomp.equals("2viewWvsR")|| _opt.typeofDecomp.equals("WvsR")){
	RMatrix_nTimeshk=new DenseDoubleMatrix2D((int) _numTok,_contextSize*(k_dim));
	RTMatrix_nTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),(int) _numTok);
	RTRMatrix_hkTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),_contextSize*(k_dim));
}	
	

if( _opt.typeofDecomp.equals("TwoStepLRvsW") ){	
	LMatrix_nTimeshk=new DenseDoubleMatrix2D((int) _numTok,_contextSize*(k_dim));
	LTMatrix_nTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),(int) _numTok);
	LTLMatrix_hkTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),_contextSize*(k_dim));
	RMatrix_nTimeshk=new DenseDoubleMatrix2D((int) _numTok,_contextSize*(k_dim));
	RTMatrix_nTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),(int) _numTok);
	RTRMatrix_hkTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),_contextSize*(k_dim));

	LTRMatrix_hkTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),_contextSize*(k_dim));
	RTLMatrix_hkTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),_contextSize*(k_dim));
}	


if(_opt.typeofDecomp.equals("2viewWvsLR")|| _opt.typeofDecomp.equals("WvsLR")){
	LMatrix_nTimeshk=new DenseDoubleMatrix2D((int) _numTok,_contextSize*(k_dim));
	LTMatrix_nTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),(int) _numTok);
	
	RMatrix_nTimeshk=new DenseDoubleMatrix2D((int) _numTok,_contextSize*(k_dim));
	RTMatrix_nTimeshk=new DenseDoubleMatrix2D(_contextSize*(k_dim),(int) _numTok);
	
	LRTLRMatrix_2hkTimes2hk=new DenseDoubleMatrix2D(2*_contextSize*(k_dim),2*_contextSize*(k_dim));

}	


int idx_doc=0;
int idx_toksAllDocs=0;
	while (idx_doc<_allDocs.size()){
		ArrayList<Integer> doc=_allDocs.get(idx_doc++);
		int idx_tok=0;
		while(idx_tok<doc.size()){
			int tok=doc.get(idx_tok);
			for(int i=1;i<=_contextSize;i++){
				
			if(!_opt.typeofDecomp.equals("2viewWvsR")&& !_opt.typeofDecomp.equals("WvsR")){	
				if (idx_tok-i>=0){
					for (int j=0;j<k_dim;j++){	
						
						LMatrix_nTimeshk.set(idx_toksAllDocs, (i-1)*(k_dim)+j, kdimEigDict.get(doc.get(idx_tok-i), j));
						LTMatrix_nTimeshk.set((i-1)*(k_dim)+j,idx_toksAllDocs, kdimEigDict.get(doc.get(idx_tok-i), j));
					}
				}
			}
			if(!_opt.typeofDecomp.equals("2viewWvsL")&& !_opt.typeofDecomp.equals("WvsL")){	
				if (idx_tok+i <doc.size()){
						for (int j=0;j<k_dim;j++){	
						
						RMatrix_nTimeshk.set(idx_toksAllDocs, (i-1)*(k_dim)+j, kdimEigDict.get(doc.get(idx_tok+i), j));
						RTMatrix_nTimeshk.set((i-1)*(k_dim)+j,idx_toksAllDocs, kdimEigDict.get(doc.get(idx_tok+i), j));
					}
				}
			}
			idx_tok++;
			idx_toksAllDocs++;
		}
		}
}		

if(_opt.typeofDecomp.equals("2viewWvsL")|| _opt.typeofDecomp.equals("WvsL") || _opt.typeofDecomp.equals("TwoStepLRvsW")){		
	LTMatrix_nTimeshk.zMult(LMatrix_nTimeshk,LTLMatrix_hkTimeshk);	
}	
	
if(_opt.typeofDecomp.equals("2viewWvsR")|| _opt.typeofDecomp.equals("WvsR") || _opt.typeofDecomp.equals("TwoStepLRvsW")){
	RTMatrix_nTimeshk.zMult(RMatrix_nTimeshk,RTRMatrix_hkTimeshk);
}
	
if(_opt.typeofDecomp.equals("TwoStepLRvsW")){
	LTMatrix_nTimeshk.zMult(RMatrix_nTimeshk,LTRMatrix_hkTimeshk);	
	RTMatrix_nTimeshk.zMult(LMatrix_nTimeshk,RTLMatrix_hkTimeshk);
}

if(_opt.typeofDecomp.equals("2viewWvsLR")|| _opt.typeofDecomp.equals("WvsLR")){
	
	LRMatrix_nTimes2hk=concatenateLR(LMatrix_nTimeshk,RMatrix_nTimeshk);
	
	LRTMatrix_2hkTimesn=concatenateLRT(LTMatrix_nTimeshk,RTMatrix_nTimeshk);
	
	LRTMatrix_2hkTimesn.zMult(LRMatrix_nTimes2hk,LRTLRMatrix_2hkTimes2hk);
}

}



	

	public FlexCompRowMatrix getLTRMatrix(){
		return LTRMatrix_hvTimeshv;
	}
	
	public FlexCompRowMatrix getRTLMatrix(){
		return RTLMatrix_hvTimeshv;
	}
	
	public FlexCompRowMatrix getLTLMatrix(){
		return LTLMatrix_hvTimeshv;
	}
	
	public FlexCompRowMatrix getRTRMatrix(){
		return RTRMatrix_hvTimeshv;
	}
	
	public FlexCompRowMatrix getWTLMatrix(){
		return WTLMatrix_vTimeshv;
	}
	
	public FlexCompRowMatrix getLTWMatrix(){
		return LTWMatrix_hvTimesv;
	}
	
	public FlexCompRowMatrix getWTRMatrix(){
		return WTRMatrix_vTimeshv;
	}
	
	public FlexCompRowMatrix getRTWMatrix(){
		return RTWMatrix_hvTimesv;
	}

	////
	public FlexCompRowMatrix  getWTLRMatrix(){
		return WTLRMatrix_vTimes2hv;
	}
	
	public FlexCompRowMatrix  getLRTWMatrix(){
		return LRTWMatrix_2hvTimesv;
	}
	
	public FlexCompRowMatrix  getLRTLRMatrix(){
		return LRTLRMatrix_2hvTimes2hv;
	}
	
	public FlexCompRowMatrix  getWTWMatrix(){
		
		return WMatrix_vTimesv;
	
	}
	
	
	/////
	public DenseDoubleMatrix2D getLTRDenseMatrix(){
		return LTRMatrix_hkTimeshk;
	}
	
	public DenseDoubleMatrix2D getRTLDenseMatrix(){
		return RTLMatrix_hkTimeshk;
	}
	
	public DenseDoubleMatrix2D getLTLDenseMatrix(){
		return LTLMatrix_hkTimeshk;
	}
	
	public DenseDoubleMatrix2D getRTRDenseMatrix(){
		return RTRMatrix_hkTimeshk;
	}
	
	public DenseDoubleMatrix2D getWTLDenseMatrix(){
		return WTLMatrix_vTimeshk;
	}
	
	public DenseDoubleMatrix2D getLTWDenseMatrix(){
		return LTWMatrix_hkTimesv;
	}
	
	public DenseDoubleMatrix2D getWTRDenseMatrix(){
		return WTRMatrix_vTimeshk;
	}
	
	public DenseDoubleMatrix2D getRTWDenseMatrix(){
		return RTWMatrix_hkTimesv;
	}

	public DenseDoubleMatrix2D  getWTLRDenseMatrix(){
		return WTLRMatrix_vTimes2hk;
	}
	
	public DenseDoubleMatrix2D  getLRTWDenseMatrix(){
		return LRTWMatrix_2hkTimesv;
	}
	
	public DenseDoubleMatrix2D  getLRTLRDenseMatrix(){
		return LRTLRMatrix_2hkTimes2hk;
	}
	
	
	/////
	
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
	
	
	public DenseDoubleMatrix2D concatenateLR(DenseDoubleMatrix2D lProjectionMatrix,
			DenseDoubleMatrix2D rProjectionMatrix) {
		DenseDoubleMatrix2D finalProjection=new DenseDoubleMatrix2D(lProjectionMatrix.rows(),(lProjectionMatrix.columns()+rProjectionMatrix.columns()));
		
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
	
	public DenseDoubleMatrix2D concatenateLRT(DenseDoubleMatrix2D lnTMatrix,
			DenseDoubleMatrix2D rnTMatrix) {
		
		DenseDoubleMatrix2D finalProjection=new DenseDoubleMatrix2D((lnTMatrix.rows()+rnTMatrix.rows()),lnTMatrix.columns());
		for (int i=0;i<lnTMatrix.rows();i++){
			for(int j=0; j<lnTMatrix.columns();j++){
				finalProjection.set(i, j, lnTMatrix.get(i, j));
				finalProjection.set(i+lnTMatrix.rows(), j, rnTMatrix.get(i, j));
			}
		}
		return finalProjection;
	}
	
	
	
	////////////////////
	public FlexCompRowMatrix concatenateLR(FlexCompRowMatrix lProjectionMatrix,
			FlexCompRowMatrix rProjectionMatrix) {
		FlexCompRowMatrix finalProjection=new FlexCompRowMatrix(lProjectionMatrix.numRows(),(lProjectionMatrix.numColumns()+rProjectionMatrix.numColumns()));
		
		for (int i=0;i<lProjectionMatrix.numRows();i++){
			for(int j=0; j<lProjectionMatrix.numColumns();j++){
				finalProjection.set(i, j, lProjectionMatrix.get(i, j));
				finalProjection.set(i, j+lProjectionMatrix.numColumns(), rProjectionMatrix.get(i, j));
			}
		}
		return finalProjection;
	}
	
	public FlexCompRowMatrix concatenateLRT(FlexCompRowMatrix lnTMatrix,
			FlexCompRowMatrix rnTMatrix) {
		
		FlexCompRowMatrix finalProjection=new FlexCompRowMatrix((lnTMatrix.numRows()+rnTMatrix.numRows()),lnTMatrix.numColumns());
		for (int i=0;i<lnTMatrix.numRows();i++){
			for(int j=0; j<lnTMatrix.numColumns();j++){
				finalProjection.set(i, j, lnTMatrix.get(i, j));
				finalProjection.set(i+lnTMatrix.numRows(), j, rnTMatrix.get(i, j));
			}
		}
		return finalProjection;
	}
	
	////
	
	
	
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
