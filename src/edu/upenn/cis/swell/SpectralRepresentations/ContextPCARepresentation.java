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
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.VectorEntry;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import no.uib.cipr.matrix.sparse.SparseVector;
import Jama.Matrix;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import edu.upenn.cis.swell.IO.Options;
import edu.upenn.cis.swell.IO.ReadDataFile;
import edu.upenn.cis.swell.MathUtils.CenterScaleNormalizeUtils;
import edu.upenn.cis.swell.MathUtils.MatrixFormatConversion;

public class ContextPCARepresentation extends SpectralRepresentation implements Serializable {

	private int _vocab_size;
	private int _contextSize,k_dim;
	ReadDataFile _rin;
	FlexCompRowMatrix WMatrix_vTimesv, LTRMatrix_hvTimeshv,RTLMatrix_hvTimeshv,
	LTLMatrix_hvTimeshv,RTRMatrix_hvTimeshv,WTLMatrix_vTimeshv,LTWMatrix_hvTimesv,WTRMatrix_vTimeshv,RTWMatrix_hvTimesv,
	 WTLRMatrix_vTimes2hv,LRTWMatrix_2hvTimesv,LRTLRMatrix_2hvTimes2hv;
	
	DenseDoubleMatrix2D WTLMatrix_vTimeshk,LTWMatrix_hkTimesv,WTRMatrix_vTimeshk,RTWMatrix_hkTimesv,
	 WTLRMatrix_vTimes2hk,LRTWMatrix_2hkTimesv,LRMatrix_nTimes2hk,LRTMatrix_2hkTimesn;
	
	Matrix kdimEigDict=null;
	
	Matrix LMatrix_nTimeshk,RMatrix_nTimeshk,LTMatrix_nTimeshk,RTMatrix_nTimeshk,
	LTRMatrix_hkTimeshk,RTLMatrix_hkTimeshk,LTLMatrix_hkTimeshk,RTRMatrix_hkTimeshk,LRTLRMatrix_2hkTimes2hk,
	LMatrix_nDocTimeshk,RMatrix_nDocTimeshk,LTMatrix_nDocTimeshk,RTMatrix_nDocTimeshk;
	
	Matrix WTLMatrix_vTimeshkAgg,WTRMatrix_vTimeshkAgg;
	
	
	FlexCompRowMatrix LMatrix_nTimeshv,RMatrix_nTimeshv,
	LMatrix_nTimesv,RMatrix_nTimesv,WMatrix_nTimesv,LTMatrix_nTimeshv,RTMatrix_nTimeshv,
	LTMatrix_nTimesv,RTMatrix_nTimesv,WTMatrix_nTimesv,LMatrix_nDocTimeshv,RMatrix_nDocTimeshv,
	WMatrix_nDocTimesv,LTMatrix_nDocTimeshv,RTMatrix_nDocTimeshv,LMatrix_nDocTimesv,RMatrix_nDocTimesv;
	
	ArrayList<Double> _smooths=new ArrayList<Double>();
	Matrix covLLAllDocsMatrix,covRRAllDocsMatrix,covLRAllDocsMatrix,covRLAllDocsMatrix,covLLMatrix,covRRMatrix,covRLMatrix,covLRMatrix;
	static final long serialVersionUID = 42L;
	int _numTok;
	ArrayList<ArrayList<Integer>> _allDocs;
	HashMap<Integer, Integer> _wordMap= null;
	
	//int idx_doc=0;
	
	
	
	
	public ContextPCARepresentation(Options opt, long numTok, ReadDataFile rin,ArrayList<ArrayList<Integer>> all_Docs) {
		super(opt, numTok);
		
		_vocab_size=super._opt.vocabSize;
		_rin=rin;
		_contextSize=_opt.contextSizeOneSide;
		_allDocs=all_Docs;	
		_numTok=(int)numTok;
		
		if(_opt.typeofDecomp.equals("LRMVL")){
		
			_smooths=opt.smoothArray;
			WTLMatrix_vTimeshkAgg=new Matrix((_vocab_size+1),_opt.smoothArray.size()*_opt.hiddenStateSize);
			WTRMatrix_vTimeshkAgg=new Matrix((_vocab_size+1),_opt.smoothArray.size()*_opt.hiddenStateSize);
			
			double[][] covLLAllDocs=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
			double[][] covRRAllDocs=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
			double[][] covLRAllDocs=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
			double[][] covRLAllDocs=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];

			covLLAllDocsMatrix= new Matrix(covLLAllDocs); 
			covRRAllDocsMatrix= new Matrix(covRRAllDocs);
			covLRAllDocsMatrix= new Matrix(covLRAllDocs);
			covRLAllDocsMatrix= new Matrix(covRLAllDocs);
			
			double[][] covLL=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
			double[][] covRR=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
			double[][] covLR=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
			double[][] covRL=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
			
			covLLMatrix= new Matrix(covLL); 
			covRRMatrix= new Matrix(covRR);
			covLRMatrix= new Matrix(covLR);
			covRLMatrix= new Matrix(covRL);
			
			
		}
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
	
	
	public Object[] processInputs(final Matrix UHat)
	throws InterruptedException, ExecutionException {

		int threads = Runtime.getRuntime().availableProcessors();
		ExecutorService service = Executors.newFixedThreadPool(threads);
		List<Future<Integer>> futures = new ArrayList<Future<Integer>>();
		final Iterator<ArrayList<Integer>> it =_allDocs.iterator();
		Object[] covs=new Object[6];
		
		
		while (it.hasNext()) {
			Callable<Integer> callable = new Callable<Integer>() {
				ArrayList<Integer> aDoc=it.next();
				
				
				
				public Integer call() throws Exception {
					Object[] covMs=null;
					covMs=computeLRMVLCovMatricesParallel(UHat, aDoc);
						updateCovs(covMs);
					return 1;
				}				
			};
			futures.add(service.submit(callable));
		}

		service.shutdown();
		List<Integer> outputs = new ArrayList<Integer>();
		int i=0;
	    for (Future<Integer> future : futures) {
	        outputs.add(future.get());
	        i++;
	        if(i%1000==0)
	        	System.out.println("===Doc Num "+(i)+" Processed===");
	    }
	    System.out.println("===Doc Processing Finished===");
	    
	    covs[0]=(Object)covLLAllDocsMatrix;
	    covs[1]=(Object)covLRAllDocsMatrix;
	    covs[2]=(Object)covRLAllDocsMatrix;
	    covs[3]=(Object)covRRAllDocsMatrix;
	    covs[4]=(Object)WTLMatrix_vTimeshkAgg;
		covs[5]=(Object)WTRMatrix_vTimeshkAgg;
		
	    return covs;  		    
	}
	

	
	public Object[] computeLRMVLCovMatrices(Matrix UHat){
		
		int idx_doc=0,_nT=0;
		DenseDoubleMatrix2D LView,RView;
		
		//Matrix UHatConcat=new Matrix(_smooths.size()*(_vocab_size+1), _opt.hiddenStateSize);
		
		
		
		Matrix WTLMatrix_vTimeshk=new Matrix((_vocab_size+1),_smooths.size()*_opt.hiddenStateSize);
		Matrix WTRMatrix_vTimeshk=new Matrix((_vocab_size+1),_smooths.size()*_opt.hiddenStateSize);
		
		//Matrix WTLMatrix_vTimeshkAgg_Aux=new Matrix((_vocab_size+1),_smooths.size()*_opt.hiddenStateSize);
		//Matrix WTRMatrix_vTimeshkAgg_Aux=new Matrix((_vocab_size+1),_smooths.size()*_opt.hiddenStateSize);
		
		
		Object[] covMatrices= new Object[6];
		//UHatConcat=concatenate(UHat,_smooths.size());
		

			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				
				//System.out.println("Proc. Doc "+idx_doc);
				if(idx_doc%1000==0)
					System.out.println("++++Proc. Doc++++"+idx_doc);
				
				//if(_allDocs.size()==1)
					//_nT=1000;
				//else
					_nT=doc.size();	
						
				//LMatrix_nDocTimeshv=new FlexCompRowMatrix(_nT,_smooths.size()*(_vocab_size+1));
				//RMatrix_nDocTimeshv=new FlexCompRowMatrix(_nT,_smooths.size()*(_vocab_size+1));
				
				//LMatrix_nDocTimesv=new FlexCompRowMatrix(_nT,(_vocab_size+1));
				//RMatrix_nDocTimesv=new FlexCompRowMatrix(_nT,(_vocab_size+1));
				
				LMatrix_nDocTimeshk=new Matrix(_nT,_smooths.size()*_opt.hiddenStateSize);
				RMatrix_nDocTimeshk=new Matrix(_nT,_smooths.size()*_opt.hiddenStateSize);
				LTMatrix_nDocTimeshk=new Matrix(_smooths.size()*_opt.hiddenStateSize,_nT);
				RTMatrix_nDocTimeshk=new Matrix(_smooths.size()*_opt.hiddenStateSize,_nT);
								
				for(int k=0;k<_smooths.size();k++){
					LMatrix_nDocTimeshk.setMatrix(0, 0,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, UHat.getMatrix(doc.get(0), doc.get(0),  0, _opt.hiddenStateSize-1));
					WTLMatrix_vTimeshk.setMatrix(doc.get(0), doc.get(0),k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, UHat.getMatrix(doc.get(0), doc.get(0),  0, _opt.hiddenStateSize-1));				
				}
				
				for(int j=1; j < _nT; j++){
					for(int k=0;k<_smooths.size();k++){
						LMatrix_nDocTimeshk.setMatrix(j,j,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, LMatrix_nDocTimeshk.getMatrix(j-1,j-1,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1).times(1-_smooths.get(k)).plus(UHat.getMatrix(doc.get(j-1), doc.get(j-1),  0, _opt.hiddenStateSize-1).times(_smooths.get(k))));
						WTLMatrix_vTimeshk.setMatrix(doc.get(j), doc.get(j), k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1,WTLMatrix_vTimeshk.getMatrix(doc.get(j), doc.get(j), k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1).plus(LMatrix_nDocTimeshk.getMatrix(j, j, k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1)));
					}
					//WTLMatrix_vTimeshkAgg.plusEquals(WTLMatrix_vTimeshkAgg_Aux);
				}
	
				for(int k=0;k<_smooths.size();k++){	
					RMatrix_nDocTimeshk.setMatrix(_nT-1, _nT-1, k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, UHat.getMatrix(doc.get(_nT-1), doc.get(_nT-1),  0, _opt.hiddenStateSize-1));	
					WTRMatrix_vTimeshk.setMatrix(doc.get(_nT-1), doc.get(_nT-1),k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, UHat.getMatrix(doc.get(_nT-1), doc.get(_nT-1),  0, _opt.hiddenStateSize-1));
				}
				//WTRMatrix_vTimeshkAgg.plusEquals(WTRMatrix_vTimeshkAgg_Aux);
				
				for(int j=_nT-2; j >=0; j--){
					for(int k=0;k<_smooths.size();k++){
						RMatrix_nDocTimeshk.setMatrix(j,j,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, RMatrix_nDocTimeshk.getMatrix(j+1,j+1,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1).times(1-_smooths.get(k)).plus(UHat.getMatrix(doc.get(j+1), doc.get(j+1),  0, _opt.hiddenStateSize-1).times(_smooths.get(k))));
						WTRMatrix_vTimeshk.setMatrix(doc.get(j), doc.get(j),k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1,WTRMatrix_vTimeshk.getMatrix(doc.get(j), doc.get(j),k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1).plus(RMatrix_nDocTimeshk.getMatrix(j, j, k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1)));
						
					
					}
					//WTRMatrix_vTimeshkAgg.plusEquals(WTRMatrix_vTimeshkAgg_Aux);
				}
				
				//LMatrix_nDocTimeshv=transform(LMatrix_nDocTimeshv);
				//RMatrix_nDocTimeshv=transform(RMatrix_nDocTimeshv);
				
				
				covLLMatrix=LMatrix_nDocTimeshk.transpose().times(LMatrix_nDocTimeshk).times(1.0/_nT);
				covLRMatrix=LMatrix_nDocTimeshk.transpose().times(RMatrix_nDocTimeshk).times(1.0/_nT);
				covRLMatrix=RMatrix_nDocTimeshk.transpose().times(LMatrix_nDocTimeshk).times(1.0/_nT);
				covRRMatrix=RMatrix_nDocTimeshk.transpose().times(RMatrix_nDocTimeshk).times(1.0/_nT);
				
				
				/*
				covLLMatrix=LMatrix_nDocTimeshk.transpose().times(LMatrix_nDocTimeshk);
				covLRMatrix=LMatrix_nDocTimeshk.transpose().times(RMatrix_nDocTimeshk);
				covRLMatrix=RMatrix_nDocTimeshk.transpose().times(LMatrix_nDocTimeshk);
				covRRMatrix=RMatrix_nDocTimeshk.transpose().times(RMatrix_nDocTimeshk);
				*/
				covLLAllDocsMatrix.plusEquals(covLLMatrix);
				covLRAllDocsMatrix.plusEquals(covLRMatrix);
				covRLAllDocsMatrix.plusEquals(covRLMatrix);
				covRRAllDocsMatrix.plusEquals(covRRMatrix);
				
			}	
			
			covMatrices[0]=(Object)covLLMatrix;
			covMatrices[1]=(Object)covRRMatrix;
			covMatrices[2]=(Object)covLRMatrix;
			covMatrices[3]=(Object)covRLMatrix;
			covMatrices[4]=(Object)WTLMatrix_vTimeshk;
			covMatrices[5]=(Object)WTRMatrix_vTimeshk;
			
			return covMatrices;
			
}

	
public void updateCovs(Object[] cvs){
	covLLAllDocsMatrix.plusEquals((Matrix) cvs[0]);
	covLRAllDocsMatrix.plusEquals((Matrix) cvs[1]);
	covRLAllDocsMatrix.plusEquals((Matrix) cvs[2]);
	covRRAllDocsMatrix.plusEquals((Matrix) cvs[3]);
	
	WTLMatrix_vTimeshkAgg.plusEquals((Matrix) cvs[4]);
	WTLMatrix_vTimeshkAgg.plusEquals((Matrix) cvs[5]);
	

}

	
public Object[] computeLRMVLCovMatricesParallel(Matrix UHat,ArrayList<Integer> doc){
		
	int _nT=0;
	
	Matrix WTLMatrix_vTimeshkDoc=new Matrix((_vocab_size+1),_smooths.size()*_opt.hiddenStateSize);
	Matrix WTRMatrix_vTimeshkDoc=new Matrix((_vocab_size+1),_smooths.size()*_opt.hiddenStateSize);
	
	Matrix LMatrix_nDocTimeshkDoc,RMatrix_nDocTimeshkDoc;
	
	Object[] covMatrices= new Object[6];

			
			_nT=doc.size();	
								
			LMatrix_nDocTimeshkDoc=new Matrix(_nT,_smooths.size()*_opt.hiddenStateSize);
			RMatrix_nDocTimeshkDoc=new Matrix(_nT,_smooths.size()*_opt.hiddenStateSize);
							
			for(int k=0;k<_smooths.size();k++){
				LMatrix_nDocTimeshkDoc.setMatrix(0, 0,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, UHat.getMatrix(doc.get(0), doc.get(0),  0, _opt.hiddenStateSize-1));
				WTLMatrix_vTimeshkDoc.setMatrix(doc.get(0), doc.get(0),k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, UHat.getMatrix(doc.get(0), doc.get(0),  0, _opt.hiddenStateSize-1));				
			}
			
			for(int j=1; j < _nT; j++){
				for(int k=0;k<_smooths.size();k++){
					LMatrix_nDocTimeshkDoc.setMatrix(j,j,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, LMatrix_nDocTimeshkDoc.getMatrix(j-1,j-1,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1).times(1-_smooths.get(k)).plus(UHat.getMatrix(doc.get(j-1), doc.get(j-1),  0, _opt.hiddenStateSize-1).times(_smooths.get(k))));
					WTLMatrix_vTimeshkDoc.setMatrix(doc.get(j), doc.get(j), k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1,WTLMatrix_vTimeshkDoc.getMatrix(doc.get(j), doc.get(j), k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1).plus(LMatrix_nDocTimeshkDoc.getMatrix(j, j, k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1)));
				}
			}

			for(int k=0;k<_smooths.size();k++){	
				RMatrix_nDocTimeshkDoc.setMatrix(_nT-1, _nT-1, k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, UHat.getMatrix(doc.get(_nT-1), doc.get(_nT-1),  0, _opt.hiddenStateSize-1));	
				WTRMatrix_vTimeshkDoc.setMatrix(doc.get(_nT-1), doc.get(_nT-1),k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, UHat.getMatrix(doc.get(_nT-1), doc.get(_nT-1),  0, _opt.hiddenStateSize-1));
			}
			
			for(int j=_nT-2; j >=0; j--){
				for(int k=0;k<_smooths.size();k++){
					RMatrix_nDocTimeshkDoc.setMatrix(j,j,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, RMatrix_nDocTimeshkDoc.getMatrix(j+1,j+1,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1).times(1-_smooths.get(k)).plus(UHat.getMatrix(doc.get(j+1), doc.get(j+1),  0, _opt.hiddenStateSize-1).times(_smooths.get(k))));
					WTRMatrix_vTimeshkDoc.setMatrix(doc.get(j), doc.get(j),k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1,WTRMatrix_vTimeshkDoc.getMatrix(doc.get(j), doc.get(j),k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1).plus(RMatrix_nDocTimeshkDoc.getMatrix(j, j, k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1)));	
				}
			}
			
			covLLMatrix=LMatrix_nDocTimeshkDoc.transpose().times(LMatrix_nDocTimeshkDoc).times(1.0/_nT);
			covLRMatrix=LMatrix_nDocTimeshkDoc.transpose().times(RMatrix_nDocTimeshkDoc).times(1.0/_nT);
			covRLMatrix=RMatrix_nDocTimeshkDoc.transpose().times(LMatrix_nDocTimeshkDoc).times(1.0/_nT);
			covRRMatrix=RMatrix_nDocTimeshkDoc.transpose().times(RMatrix_nDocTimeshkDoc).times(1.0/_nT);
			
			
		
		covMatrices[0]=(Object)covLLMatrix;
		covMatrices[1]=(Object)covRRMatrix;
		covMatrices[2]=(Object)covLRMatrix;
		covMatrices[3]=(Object)covRLMatrix;
		covMatrices[4]=(Object)WTLMatrix_vTimeshkDoc;
		covMatrices[5]=(Object)WTRMatrix_vTimeshkDoc;
		
		return covMatrices;
			
}
	
	
	
	
	public Object[] computeLRMVLCovMatricesOld2(Matrix UHat, ArrayList<Integer> doc){
				
				Object[] covMatrices= new Object[4];
				int _nT;
				DenseDoubleMatrix2D viewAux1;
				FlexCompRowMatrix auxM1;
				Matrix LMatrix_nDocTimeshk,RMatrix_nDocTimeshk;
				FlexCompRowMatrix LMatrix_nDocTimeshv,RMatrix_nDocTimeshv,LMatrix_nDocTimesv,RMatrix_nDocTimesv;
				
				DenseDoubleMatrix2D LView,RView;
				
				
				_nT=doc.size();	
						
				LMatrix_nDocTimeshv=new FlexCompRowMatrix(_nT,_smooths.size()*(_vocab_size+1));
				RMatrix_nDocTimeshv=new FlexCompRowMatrix(_nT,_smooths.size()*(_vocab_size+1));
				
				LMatrix_nDocTimesv=new FlexCompRowMatrix(_nT,(_vocab_size+1));
				RMatrix_nDocTimesv=new FlexCompRowMatrix(_nT,(_vocab_size+1));
				
				LMatrix_nDocTimeshk=new Matrix(_nT,_smooths.size()*_opt.hiddenStateSize);
				RMatrix_nDocTimeshk=new Matrix(_nT,_smooths.size()*_opt.hiddenStateSize);
				
				viewAux1=new DenseDoubleMatrix2D(_nT,_opt.hiddenStateSize);
				auxM1=new FlexCompRowMatrix(_nT,(_vocab_size+1));

						
				LView=new DenseDoubleMatrix2D(_nT,_smooths.size()*_opt.hiddenStateSize);
				RView=new DenseDoubleMatrix2D(_nT,_smooths.size()*_opt.hiddenStateSize);
				
				for(int k=0;k<_smooths.size();k++){
					LMatrix_nDocTimeshv.set(0,(k+1)*doc.get(0),1);
				}
				LMatrix_nDocTimesv.set(0,doc.get(0),1);
				for(int j=1; j < _nT; j++){
						LMatrix_nDocTimesv.set(j,doc.get(j-1),1);
						Iterator<VectorEntry> it =LMatrix_nDocTimesv.getRow(j-1).iterator();
						while(it.hasNext()){
							VectorEntry ment = it.next();
							for(int m=0;m<_smooths.size();m++){
								LMatrix_nDocTimeshv.set(j,(m+1)*ment.index(), ment.get()*(1-_smooths.get(m)));
							}
						}
						for(int m=0;m<_smooths.size();m++){
						LMatrix_nDocTimeshv.add(j,(m+1)*doc.get(j-1), _smooths.get(m));	
					}
				}
				//
				for(int k=0;k<_smooths.size();k++){
					RMatrix_nDocTimeshv.set(0,(k+1)*doc.get(_nT-1),1);
					
				}
				RMatrix_nDocTimesv.set(0,doc.get(_nT-1),1);
				int idx=1;
				for(int j=_nT-2; j >=0; j--){
						RMatrix_nDocTimesv.set(idx,doc.get(j+1),1);
						Iterator<VectorEntry> it =RMatrix_nDocTimesv.getRow(idx-1).iterator();
						while(it.hasNext()){
							VectorEntry ment = it.next();
							for(int m=0;m<_smooths.size();m++){
								RMatrix_nDocTimeshv.set(idx,(m+1)*ment.index(), ment.get()*(1-_smooths.get(m)));
										
							}
						}
						for(int m=0;m<_smooths.size();m++){
						RMatrix_nDocTimeshv.add(idx,(m+1)*doc.get(j+1), _smooths.get(m));
					}
						idx++;
				}
				LMatrix_nDocTimeshv=transform(LMatrix_nDocTimeshv);
				RMatrix_nDocTimeshv=transform(RMatrix_nDocTimeshv);
				
				for(int z=0;z<_smooths.size();z++){
						
					for(int i=0; i<LMatrix_nDocTimeshv.numRows();i++ ){
						Iterator<VectorEntry> it =LMatrix_nDocTimeshv.getRow(i).iterator();
						while(it.hasNext()){
							VectorEntry ment = it.next();
							if(ment.index() >= z*(_vocab_size+1) && ment.index() < (z+1)*(_vocab_size+1) )
								auxM1.set(i, ment.index()-(z*(_vocab_size+1)),  ment.get());
						}
					}
					
					MatrixFormatConversion.createSparseMatrixCOLT(auxM1).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat), viewAux1);
					
					for(int i=0; i<viewAux1.rows();i++ ){
						for(int j=z*_opt.hiddenStateSize; j <(z+1)*(_opt.hiddenStateSize);j++){
							LView.set(i,j,  viewAux1.get(i,j-(z*(_opt.hiddenStateSize))));
						}
					}
					
					
					for(int i=0; i<RMatrix_nDocTimeshv.numRows();i++ ){
						Iterator<VectorEntry> it =RMatrix_nDocTimeshv.getRow(i).iterator();
						while(it.hasNext()){
							VectorEntry ment = it.next();
							if(ment.index() >= z*(_vocab_size+1) && ment.index() < (z+1)*(_vocab_size+1) )
								auxM1.set(i, ment.index()-(z*(_vocab_size+1)),  ment.get());
						}
					}

					MatrixFormatConversion.createSparseMatrixCOLT(auxM1).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat), viewAux1); 
					
					for(int i=0; i<viewAux1.rows();i++ ){
						for(int j=z*_opt.hiddenStateSize; j <(z+1)*(_opt.hiddenStateSize);j++){
							RView.set(i,j,  viewAux1.get(i,j-(z*(_opt.hiddenStateSize))));
						}
					}
					
					
				}
				LMatrix_nDocTimeshk=MatrixFormatConversion.createDenseMatrixJAMA(LView);
				RMatrix_nDocTimeshk=MatrixFormatConversion.createDenseMatrixJAMA(RView);
				
				covLLMatrix=LMatrix_nDocTimeshk.transpose().times(LMatrix_nDocTimeshk);
				covLRMatrix=LMatrix_nDocTimeshk.transpose().times(RMatrix_nDocTimeshk);
				covRLMatrix=RMatrix_nDocTimeshk.transpose().times(LMatrix_nDocTimeshk);
				covRRMatrix=RMatrix_nDocTimeshk.transpose().times(RMatrix_nDocTimeshk);
					
				covMatrices[0]=(Object)covLLMatrix;
				covMatrices[1]=(Object)covRRMatrix;
				covMatrices[2]=(Object)covLRMatrix;
				covMatrices[3]=(Object)covRLMatrix;
			
			return covMatrices;
			
	}
		
	
	public Object[] computeAggMatrices(Matrix UHat){
		
		
		int idx_doc=0,_nT=0;
		
		FlexCompRowMatrix WTLMatrix_vTimeshvAgg=new FlexCompRowMatrix((_vocab_size+1),_opt.smoothArray.size()*(_vocab_size+1));
		FlexCompRowMatrix WTRMatrix_vTimeshvAgg=new FlexCompRowMatrix((_vocab_size+1),_opt.smoothArray.size()*(_vocab_size+1));
		
		DenseDoubleMatrix2D WTLMatrix_vTimeshkAggCOLT=new DenseDoubleMatrix2D((_vocab_size+1),_opt.smoothArray.size()*_opt.hiddenStateSize);
		DenseDoubleMatrix2D WTRMatrix_vTimeshkAggCOLT=new DenseDoubleMatrix2D((_vocab_size+1),_opt.smoothArray.size()*_opt.hiddenStateSize);
		
		
		Matrix WTLMatrix_vTimeshkAgg=new Matrix((_vocab_size+1),_opt.smoothArray.size()*_opt.hiddenStateSize);
		Matrix WTRMatrix_vTimeshkAgg=new Matrix((_vocab_size+1),_opt.smoothArray.size()*_opt.hiddenStateSize);
		
		
		Object[] covMatrices= new Object[2];
		//UHatConcat=concatenate(UHat,_smooths.size());
		DenseDoubleMatrix2D viewAux2;
		FlexCompRowMatrix auxM2;
		viewAux2=new DenseDoubleMatrix2D((_vocab_size+1),_opt.hiddenStateSize);
		auxM2=new FlexCompRowMatrix((_vocab_size+1),(_vocab_size+1));


			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				
				if(idx_doc%1000==0)
					System.out.println("++++Proc. Doc "+idx_doc);
				
				_nT=doc.size();	
						
				LMatrix_nDocTimesv=new FlexCompRowMatrix(_nT,(_vocab_size+1));
				RMatrix_nDocTimesv=new FlexCompRowMatrix(_nT,(_vocab_size+1));
										
				for(int k=0;k<_smooths.size();k++){
					WTLMatrix_vTimeshvAgg.set(doc.get(0),(k+1)*doc.get(0),1);
				}
				LMatrix_nDocTimesv.set(0,doc.get(0),1);
				for(int j=1; j < _nT; j++){
						LMatrix_nDocTimesv.set(j,doc.get(j-1),1);
						Iterator<VectorEntry> it =LMatrix_nDocTimesv.getRow(j-1).iterator();
						while(it.hasNext()){
							VectorEntry ment = it.next();
							for(int m=0;m<_smooths.size();m++){
								WTLMatrix_vTimeshvAgg.set(doc.get(j),(m+1)*ment.index(), ment.get()*(1-_smooths.get(m)));
							}
						}
						for(int m=0;m<_smooths.size();m++){
						WTLMatrix_vTimeshvAgg.add(doc.get(j),(m+1)*doc.get(j-1), _smooths.get(m));
					}
				}
				//
				for(int k=0;k<_smooths.size();k++){
					WTRMatrix_vTimeshvAgg.set(doc.get(_nT-1),(k+1)*doc.get(_nT-1),1);	
				}
				RMatrix_nDocTimesv.set(0,doc.get(_nT-1),1);
				int idx=1;
				for(int j=_nT-2; j >=0; j--){
						RMatrix_nDocTimesv.set(idx,doc.get(j+1),1);
						Iterator<VectorEntry> it =RMatrix_nDocTimesv.getRow(idx-1).iterator();
						while(it.hasNext()){
							VectorEntry ment = it.next();
							for(int m=0;m<_smooths.size();m++){
								WTRMatrix_vTimeshvAgg.set(doc.get(j),(m+1)*ment.index(), ment.get()*(1-_smooths.get(m)));	
							}
						}
						for(int m=0;m<_smooths.size();m++){
						WTRMatrix_vTimeshvAgg.add(doc.get(j),(m+1)*doc.get(j+1), _smooths.get(m));
					}
				idx++;
				}
			}	
			
			WTLMatrix_vTimeshvAgg=transform(WTLMatrix_vTimeshvAgg);
			WTRMatrix_vTimeshvAgg=transform(WTRMatrix_vTimeshvAgg);
			
			for(int z=0;z<_smooths.size();z++){
				
				
				for(int i=0; i<WTLMatrix_vTimeshvAgg.numRows();i++ ){
					Iterator<VectorEntry> it =WTLMatrix_vTimeshvAgg.getRow(i).iterator();
					while(it.hasNext()){
						VectorEntry ment = it.next();
						if(ment.index() >= z*(_vocab_size+1) && ment.index() < (z+1)*(_vocab_size+1) )
							auxM2.set(i, ment.index()-(z*(_vocab_size+1)),  ment.get());
					}
				}
				
				MatrixFormatConversion.createSparseMatrixCOLT(auxM2).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat),viewAux2);
				
				for(int i=0; i<viewAux2.rows();i++ ){
					for(int j=z*_opt.hiddenStateSize; j <(z+1)*(_opt.hiddenStateSize);j++){
						WTLMatrix_vTimeshkAggCOLT.set(i,j,  viewAux2.get(i,j-(z*(_opt.hiddenStateSize))));
					}
				}
				
				for(int i=0; i<WTRMatrix_vTimeshvAgg.numRows();i++ ){
					Iterator<VectorEntry> it =WTRMatrix_vTimeshvAgg.getRow(i).iterator();
					while(it.hasNext()){
						VectorEntry ment = it.next();
						if(ment.index() >= z*(_vocab_size+1) && ment.index() < (z+1)*(_vocab_size+1) )
							auxM2.set(i, ment.index()-(z*(_vocab_size+1)),  ment.get());
					}
				}
				
				MatrixFormatConversion.createSparseMatrixCOLT(auxM2).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat),viewAux2);
				
				for(int i=0; i<viewAux2.rows();i++ ){
					for(int j=z*_opt.hiddenStateSize; j <(z+1)*(_opt.hiddenStateSize);j++){
						WTRMatrix_vTimeshkAggCOLT.set(i,j,  viewAux2.get(i,j-(z*(_opt.hiddenStateSize))));
					}
				}
				
			}
			
			WTLMatrix_vTimeshkAgg=MatrixFormatConversion.createDenseMatrixJAMA(WTLMatrix_vTimeshkAggCOLT);
			WTRMatrix_vTimeshkAgg=MatrixFormatConversion.createDenseMatrixJAMA(WTRMatrix_vTimeshkAggCOLT);
			
			covMatrices[0]=(Object)WTLMatrix_vTimeshkAgg;
			covMatrices[1]=(Object)WTRMatrix_vTimeshkAgg;
			
			return covMatrices;
	}
	
	
	public Object[] computeLRMVLCovMatricesOld(Matrix UHat){
		
	
	int idx_doc=0,_nT=0;
	DenseDoubleMatrix2D LView,RView;
	
	//Matrix UHatConcat=new Matrix(_smooths.size()*(_vocab_size+1), _opt.hiddenStateSize);
	
	FlexCompRowMatrix WTLMatrix_vTimeshvAgg=new FlexCompRowMatrix((_vocab_size+1),_smooths.size()*(_vocab_size+1));
	FlexCompRowMatrix WTRMatrix_vTimeshvAgg=new FlexCompRowMatrix((_vocab_size+1),_smooths.size()*(_vocab_size+1));
	
	DenseDoubleMatrix2D WTLMatrix_vTimeshkAggCOLT=new DenseDoubleMatrix2D((_vocab_size+1),_smooths.size()*_opt.hiddenStateSize);
	DenseDoubleMatrix2D WTRMatrix_vTimeshkAggCOLT=new DenseDoubleMatrix2D((_vocab_size+1),_smooths.size()*_opt.hiddenStateSize);
	
	
	Matrix WTLMatrix_vTimeshkAgg=new Matrix((_vocab_size+1),_smooths.size()*_opt.hiddenStateSize);
	Matrix WTRMatrix_vTimeshkAgg=new Matrix((_vocab_size+1),_smooths.size()*_opt.hiddenStateSize);
	
	
	Object[] covMatrices= new Object[6];
	//UHatConcat=concatenate(UHat,_smooths.size());
	DenseDoubleMatrix2D viewAux1,viewAux2;
	FlexCompRowMatrix auxM1,auxM2;
	viewAux2=new DenseDoubleMatrix2D((_vocab_size+1),_opt.hiddenStateSize);
	auxM2=new FlexCompRowMatrix((_vocab_size+1),(_vocab_size+1));


		while (idx_doc<_allDocs.size()){
			ArrayList<Integer> doc=_allDocs.get(idx_doc++);
			
			if(idx_doc%1000==0)
				System.out.println("++++Proc. Doc "+idx_doc);
			
			if(_allDocs.size()==1)
				_nT=1000;
			else
				_nT=doc.size();	
					
			LMatrix_nDocTimeshv=new FlexCompRowMatrix(_nT,_smooths.size()*(_vocab_size+1));
			RMatrix_nDocTimeshv=new FlexCompRowMatrix(_nT,_smooths.size()*(_vocab_size+1));
			
			LMatrix_nDocTimesv=new FlexCompRowMatrix(_nT,(_vocab_size+1));
			RMatrix_nDocTimesv=new FlexCompRowMatrix(_nT,(_vocab_size+1));
			
			LMatrix_nDocTimeshk=new Matrix(_nT,_smooths.size()*_opt.hiddenStateSize);
			RMatrix_nDocTimeshk=new Matrix(_nT,_smooths.size()*_opt.hiddenStateSize);
			LTMatrix_nDocTimeshk=new Matrix(_smooths.size()*_opt.hiddenStateSize,_nT);
			RTMatrix_nDocTimeshk=new Matrix(_smooths.size()*_opt.hiddenStateSize,_nT);
			
			viewAux1=new DenseDoubleMatrix2D(_nT,_opt.hiddenStateSize);
			auxM1=new FlexCompRowMatrix(_nT,(_vocab_size+1));

					
			LView=new DenseDoubleMatrix2D(_nT,_smooths.size()*_opt.hiddenStateSize);
			RView=new DenseDoubleMatrix2D(_nT,_smooths.size()*_opt.hiddenStateSize);
			
			for(int k=0;k<_smooths.size();k++){
				LMatrix_nDocTimeshv.set(0,(k+1)*doc.get(0),1);
				WTLMatrix_vTimeshvAgg.set(doc.get(0),(k+1)*doc.get(0),1);
			}
			LMatrix_nDocTimesv.set(0,doc.get(0),1);
			for(int j=1; j < _nT; j++){
					LMatrix_nDocTimesv.set(j,doc.get(j-1),1);
					Iterator<VectorEntry> it =LMatrix_nDocTimesv.getRow(j-1).iterator();
					while(it.hasNext()){
						VectorEntry ment = it.next();
						for(int m=0;m<_smooths.size();m++){
							LMatrix_nDocTimeshv.set(j,(m+1)*ment.index(), ment.get()*(1-_smooths.get(m)));
							WTLMatrix_vTimeshvAgg.set(doc.get(j),(m+1)*ment.index(), ment.get()*(1-_smooths.get(m)));
						}
					}
					for(int m=0;m<_smooths.size();m++){
					LMatrix_nDocTimeshv.add(j,(m+1)*doc.get(j-1), _smooths.get(m));
					
					WTLMatrix_vTimeshvAgg.add(doc.get(j),(m+1)*doc.get(j-1), _smooths.get(m));
				}
			}
			//
			for(int k=0;k<_smooths.size();k++){
				RMatrix_nDocTimeshv.set(0,(k+1)*doc.get(_nT-1),1);
				WTRMatrix_vTimeshvAgg.set(doc.get(_nT-1),(k+1)*doc.get(_nT-1),1);
				
			}
			RMatrix_nDocTimesv.set(0,doc.get(_nT-1),1);
			int idx=1;
			for(int j=_nT-2; j >=0; j--){
					RMatrix_nDocTimesv.set(idx,doc.get(j+1),1);
					Iterator<VectorEntry> it =RMatrix_nDocTimesv.getRow(idx-1).iterator();
					while(it.hasNext()){
						VectorEntry ment = it.next();
						for(int m=0;m<_smooths.size();m++){
							RMatrix_nDocTimeshv.set(idx,(m+1)*ment.index(), ment.get()*(1-_smooths.get(m)));
							
							WTRMatrix_vTimeshvAgg.set(doc.get(j),(m+1)*ment.index(), ment.get()*(1-_smooths.get(m)));
							
						}
					}
					for(int m=0;m<_smooths.size();m++){
					RMatrix_nDocTimeshv.add(idx,(m+1)*doc.get(j+1), _smooths.get(m));
					
					WTRMatrix_vTimeshvAgg.add(doc.get(j),(m+1)*doc.get(j+1), _smooths.get(m));
				}
					idx++;
			}
			LMatrix_nDocTimeshv=transform(LMatrix_nDocTimeshv);
			RMatrix_nDocTimeshv=transform(RMatrix_nDocTimeshv);
			
			for(int z=0;z<_smooths.size();z++){
					
				for(int i=0; i<LMatrix_nDocTimeshv.numRows();i++ ){
					Iterator<VectorEntry> it =LMatrix_nDocTimeshv.getRow(i).iterator();
					while(it.hasNext()){
						VectorEntry ment = it.next();
						if(ment.index() >= z*(_vocab_size+1) && ment.index() < (z+1)*(_vocab_size+1) )
							auxM1.set(i, ment.index()-(z*(_vocab_size+1)),  ment.get());
					}
				}
				
				MatrixFormatConversion.createSparseMatrixCOLT(auxM1).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat), viewAux1);
				
				for(int i=0; i<viewAux1.rows();i++ ){
					for(int j=z*_opt.hiddenStateSize; j <(z+1)*(_opt.hiddenStateSize);j++){
						LView.set(i,j,  viewAux1.get(i,j-(z*(_opt.hiddenStateSize))));
					}
				}
				
				
				for(int i=0; i<RMatrix_nDocTimeshv.numRows();i++ ){
					Iterator<VectorEntry> it =RMatrix_nDocTimeshv.getRow(i).iterator();
					while(it.hasNext()){
						VectorEntry ment = it.next();
						if(ment.index() >= z*(_vocab_size+1) && ment.index() < (z+1)*(_vocab_size+1) )
							auxM1.set(i, ment.index()-(z*(_vocab_size+1)),  ment.get());
					}
				}

				MatrixFormatConversion.createSparseMatrixCOLT(auxM1).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat), viewAux1); 
				
				for(int i=0; i<viewAux1.rows();i++ ){
					for(int j=z*_opt.hiddenStateSize; j <(z+1)*(_opt.hiddenStateSize);j++){
						RView.set(i,j,  viewAux1.get(i,j-(z*(_opt.hiddenStateSize))));
					}
				}
				
				
			}
			LMatrix_nDocTimeshk=MatrixFormatConversion.createDenseMatrixJAMA(LView);
			RMatrix_nDocTimeshk=MatrixFormatConversion.createDenseMatrixJAMA(RView);
			
			/*
			covLLMatrix=LMatrix_nDocTimeshk.transpose().times(LMatrix_nDocTimeshk).times(1/_nT);
			covLRMatrix=LMatrix_nDocTimeshk.transpose().times(RMatrix_nDocTimeshk).times(1/_nT);
			covRLMatrix=RMatrix_nDocTimeshk.transpose().times(LMatrix_nDocTimeshk).times(1/_nT);
			covRRMatrix=RMatrix_nDocTimeshk.transpose().times(RMatrix_nDocTimeshk).times(1/_nT);
			*/
			
			covLLMatrix=LMatrix_nDocTimeshk.transpose().times(LMatrix_nDocTimeshk);
			covLRMatrix=LMatrix_nDocTimeshk.transpose().times(RMatrix_nDocTimeshk);
			covRLMatrix=RMatrix_nDocTimeshk.transpose().times(LMatrix_nDocTimeshk);
			covRRMatrix=RMatrix_nDocTimeshk.transpose().times(RMatrix_nDocTimeshk);
			
			
			
			covLLAllDocsMatrix.plusEquals(covLLMatrix);
			covLRAllDocsMatrix.plusEquals(covLRMatrix);
			covRLAllDocsMatrix.plusEquals(covRLMatrix);
			covRRAllDocsMatrix.plusEquals(covRRMatrix);
			
		}	
		
		WTLMatrix_vTimeshvAgg=transform(WTLMatrix_vTimeshvAgg);
		WTRMatrix_vTimeshvAgg=transform(WTRMatrix_vTimeshvAgg);
		
		
	   
		for(int z=0;z<_smooths.size();z++){
			
			
			for(int i=0; i<WTLMatrix_vTimeshvAgg.numRows();i++ ){
				Iterator<VectorEntry> it =WTLMatrix_vTimeshvAgg.getRow(i).iterator();
				while(it.hasNext()){
					VectorEntry ment = it.next();
					if(ment.index() >= z*(_vocab_size+1) && ment.index() < (z+1)*(_vocab_size+1) )
						auxM2.set(i, ment.index()-(z*(_vocab_size+1)),  ment.get());
				}
			}
			
			MatrixFormatConversion.createSparseMatrixCOLT(auxM2).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat),viewAux2);
			
			for(int i=0; i<viewAux2.rows();i++ ){
				for(int j=z*_opt.hiddenStateSize; j <(z+1)*(_opt.hiddenStateSize);j++){
					WTLMatrix_vTimeshkAggCOLT.set(i,j,  viewAux2.get(i,j-(z*(_opt.hiddenStateSize))));
				}
			}
			
			
			
			for(int i=0; i<WTRMatrix_vTimeshvAgg.numRows();i++ ){
				Iterator<VectorEntry> it =WTRMatrix_vTimeshvAgg.getRow(i).iterator();
				while(it.hasNext()){
					VectorEntry ment = it.next();
					if(ment.index() >= z*(_vocab_size+1) && ment.index() < (z+1)*(_vocab_size+1) )
						auxM2.set(i, ment.index()-(z*(_vocab_size+1)),  ment.get());
				}
			}
			
			MatrixFormatConversion.createSparseMatrixCOLT(auxM2).zMult(MatrixFormatConversion.createDenseMatrixCOLT(UHat),viewAux2);
			
			for(int i=0; i<viewAux2.rows();i++ ){
				for(int j=z*_opt.hiddenStateSize; j <(z+1)*(_opt.hiddenStateSize);j++){
					WTRMatrix_vTimeshkAggCOLT.set(i,j,  viewAux2.get(i,j-(z*(_opt.hiddenStateSize))));
				}
			}
			
			
		}
		
		
		WTLMatrix_vTimeshkAgg=MatrixFormatConversion.createDenseMatrixJAMA(WTLMatrix_vTimeshkAggCOLT);
		WTRMatrix_vTimeshkAgg=MatrixFormatConversion.createDenseMatrixJAMA(WTRMatrix_vTimeshkAggCOLT);
		
		
		covMatrices[0]=(Object)covLLMatrix;
		covMatrices[1]=(Object)covRRMatrix;
		covMatrices[2]=(Object)covLRMatrix;
		covMatrices[3]=(Object)covRLMatrix;
		covMatrices[4]=(Object)WTLMatrix_vTimeshkAgg;
		covMatrices[5]=(Object)WTRMatrix_vTimeshkAgg;
		
		return covMatrices;
		
}
	
	
	public Object[] computeLRMVLCovMatricesTrain(Matrix UHat){
		
		
		int idx_doc=0,_nT=0,c=0;
		
		
		Matrix LMatrix_nTimeshk=null,RMatrix_nTimeshk=null,LMatrix_nTimeshkAll=null,RMatrix_nTimeshkAll=null;
		Object[] covMatrices= new Object[6];
		//UHatConcat=concatenate(UHat,_smooths.size());
		

			while (idx_doc<_allDocs.size()){
				ArrayList<Integer> doc=_allDocs.get(idx_doc++);
				
				System.out.println("Proc. Doc "+idx_doc);
				if(idx_doc%1000==0)
					System.out.println("++++Proc. Doc "+idx_doc);
				
					_nT=doc.size();	
						
				LMatrix_nTimeshk=new Matrix(_nT,_smooths.size()*_opt.hiddenStateSize);
				RMatrix_nTimeshk=new Matrix(_nT,_smooths.size()*_opt.hiddenStateSize);
								
				for(int k=0;k<_smooths.size();k++){
					LMatrix_nTimeshk.setMatrix(0, 0,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, UHat.getMatrix(doc.get(0), doc.get(0),  0, _opt.hiddenStateSize-1));
				}
				
				for(int j=1; j < _nT; j++){
					for(int k=0;k<_smooths.size();k++){
						LMatrix_nTimeshk.setMatrix(j,j,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, LMatrix_nTimeshk.getMatrix(j-1,j-1,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1).times(1-_smooths.get(k)).plus(UHat.getMatrix(doc.get(j-1), doc.get(j-1),  0, _opt.hiddenStateSize-1).times(_smooths.get(k))));
					}
				}
	
				for(int k=0;k<_smooths.size();k++){	
					RMatrix_nTimeshk.setMatrix(_nT-1, _nT-1, k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, UHat.getMatrix(doc.get(_nT-1), doc.get(_nT-1),  0, _opt.hiddenStateSize-1));	
				}
				
				for(int j=_nT-2; j >=0; j--){
					for(int k=0;k<_smooths.size();k++){
						RMatrix_nTimeshk.setMatrix(j,j,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1, RMatrix_nTimeshk.getMatrix(j+1,j+1,k*(_opt.hiddenStateSize),(k+1)*(_opt.hiddenStateSize)-1).times(1-_smooths.get(k)).plus(UHat.getMatrix(doc.get(j+1), doc.get(j+1),  0, _opt.hiddenStateSize-1).times(_smooths.get(k))));		
					}
			}
				
				if (c==0){
					LMatrix_nTimeshkAll=LMatrix_nTimeshk;
					RMatrix_nTimeshkAll=RMatrix_nTimeshk;
					c++;
				}
				else{
					LMatrix_nTimeshkAll= concatenateLRT(LMatrix_nTimeshkAll,LMatrix_nTimeshk);
					RMatrix_nTimeshkAll= concatenateLRT(RMatrix_nTimeshkAll,RMatrix_nTimeshk);
				}
			
			}	
				
			
			covMatrices[0]=(Object)LMatrix_nTimeshkAll;
			covMatrices[1]=(Object)RMatrix_nTimeshkAll;
			
			return covMatrices;
			
	}
	
		public void computeTrainLRMatrices(){
	
			LMatrix_nTimeshv=new FlexCompRowMatrix((int) _numTok,_contextSize*(_vocab_size+1));
			RMatrix_nTimeshv=new FlexCompRowMatrix((int) _numTok,_contextSize*(_vocab_size+1));
			LTMatrix_nTimeshv=new FlexCompRowMatrix(_contextSize*(_vocab_size+1),(int) _numTok);
			RTMatrix_nTimeshv=new FlexCompRowMatrix(_contextSize*(_vocab_size+1),(int) _numTok);
		//}
		WMatrix_nTimesv=new FlexCompRowMatrix((int) _numTok,(_vocab_size+1));
		WTMatrix_nTimesv=new FlexCompRowMatrix((_vocab_size+1),(int) _numTok);
		
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
		
		if( _opt.typeofDecomp.equals("TwoStepLRvsW") ||  _opt.typeofDecomp.equals("LRMVL1") ){
		
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
	
	
	
	public void computeWTWMatrix(){
		
	//Required for LRMVL. We need not compute all the other matrices.
		
		WMatrix_vTimesv=new FlexCompRowMatrix((_vocab_size+1),_vocab_size+1);
			
		int idx_doc=0;
		System.out.println("+++Entering loop over the documents+++");
		while (idx_doc<_allDocs.size()){
			ArrayList<Integer> doc=_allDocs.get(idx_doc++);
			int idx_tok=0;
			
			while(idx_tok<doc.size()){
				int tok=doc.get(idx_tok);
				WMatrix_vTimesv.add(tok, tok, 1);
				
				idx_tok++;
			}
			if(idx_doc%1000==0)
				System.out.println("+Doc Num: "+idx_doc+" Processed");
		}
	
	
				
	}
	
	
	public void transformMatrices(){
		
		//We can not have n*hv sparse matrices due to limits on max. matrix sizes so we will have to perform the multiplication here only.
		
	 
		if( _opt.typeofDecomp.equals("TwoStepLRvsW") || _opt.typeofDecomp.equals("LRMVL1") ){
		
			LTRMatrix_hvTimeshv=transform(LTRMatrix_hvTimeshv);
			RTLMatrix_hvTimeshv=transform(RTLMatrix_hvTimeshv);
			LTLMatrix_hvTimeshv=transform(LTLMatrix_hvTimeshv);
			RTRMatrix_hvTimeshv=transform(RTRMatrix_hvTimeshv);
			WTLMatrix_vTimeshv=transform(WTLMatrix_vTimeshv);
			LTWMatrix_hvTimesv=transform(LTWMatrix_hvTimesv);
			WMatrix_vTimesv=transform(WMatrix_vTimesv);
			WTRMatrix_vTimeshv=transform(WTRMatrix_vTimeshv);
			RTWMatrix_hvTimesv=transform(RTWMatrix_hvTimesv);
						
	}
		
		if(_opt.typeofDecomp.equals("2viewWvsL")|| _opt.typeofDecomp.equals("WvsL")){
			
			WTLMatrix_vTimeshv=transform(WTLMatrix_vTimeshv);
			LTWMatrix_hvTimesv= transform(LTWMatrix_hvTimesv);
			LTLMatrix_hvTimeshv= transform(LTLMatrix_hvTimeshv);
			WMatrix_vTimesv= transform(WMatrix_vTimesv);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsR")|| _opt.typeofDecomp.equals("WvsR") ){
			
			WTRMatrix_vTimeshv=transform(WTRMatrix_vTimeshv);
			RTWMatrix_hvTimesv= transform(RTWMatrix_hvTimesv);
			RTRMatrix_hvTimeshv= transform(RTRMatrix_hvTimeshv);
			WMatrix_vTimesv= transform(WMatrix_vTimesv);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsLR")|| _opt.typeofDecomp.equals("WvsLR")){
			
			WTLRMatrix_vTimes2hv=transform(WTLRMatrix_vTimes2hv);
			LRTWMatrix_2hvTimesv=transform(LRTWMatrix_2hvTimesv);
			LRTLRMatrix_2hvTimes2hv=transform(LRTLRMatrix_2hvTimes2hv);
			WMatrix_vTimesv=transform(WMatrix_vTimesv);
		}
				
	}	
	
	
public void transformWTWMatrix(){
			WMatrix_vTimesv=transform(WMatrix_vTimesv);				
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
		LTLMatrix_hkTimeshk=new Matrix(_contextSize*(k_dim),_contextSize*(k_dim));
	}

	if(_opt.typeofDecomp.equals("2viewWvsR")|| _opt.typeofDecomp.equals("WvsR")){
		RTRMatrix_hkTimeshk=new Matrix(_contextSize*(k_dim),_contextSize*(k_dim));
	}	
		
	if( _opt.typeofDecomp.equals("TwoStepLRvsW") ){	
		LTLMatrix_hkTimeshk=new Matrix(_contextSize*(k_dim),_contextSize*(k_dim));
		RTRMatrix_hkTimeshk=new Matrix(_contextSize*(k_dim),_contextSize*(k_dim));
		LTRMatrix_hkTimeshk=new Matrix(_contextSize*(k_dim),_contextSize*(k_dim));
		RTLMatrix_hkTimeshk=new Matrix(_contextSize*(k_dim),_contextSize*(k_dim));
	}	

	if(_opt.typeofDecomp.equals("2viewWvsLR")|| _opt.typeofDecomp.equals("WvsLR")){		
		
		LRTLRMatrix_2hkTimes2hk=new Matrix(2*_contextSize*(k_dim),2*_contextSize*(k_dim));

	}		
	
	
int idx_doc=0;
	while (idx_doc<_allDocs.size()){
		ArrayList<Integer> doc=_allDocs.get(idx_doc++);
		int idx_tok=0;
		int tokSize=doc.size();
		////////////////////////
		

		if(_opt.typeofDecomp.equals("2viewWvsL")|| _opt.typeofDecomp.equals("WvsL")){	
			LMatrix_nTimeshk=new Matrix(tokSize,_contextSize*(k_dim));
			LTMatrix_nTimeshk=new Matrix(_contextSize*(k_dim),tokSize);
		}

		if(_opt.typeofDecomp.equals("2viewWvsR")|| _opt.typeofDecomp.equals("WvsR")){
			RMatrix_nTimeshk=new Matrix(tokSize,_contextSize*(k_dim));
			RTMatrix_nTimeshk=new Matrix(_contextSize*(k_dim),tokSize);
		}	
			

		if( _opt.typeofDecomp.equals("TwoStepLRvsW") ){	
			LMatrix_nTimeshk=new Matrix(tokSize,_contextSize*(k_dim));
			LTMatrix_nTimeshk=new Matrix(_contextSize*(k_dim),tokSize);
			RMatrix_nTimeshk=new Matrix(tokSize,_contextSize*(k_dim));
			RTMatrix_nTimeshk=new Matrix(_contextSize*(k_dim),tokSize);
		}	


		if(_opt.typeofDecomp.equals("2viewWvsLR")|| _opt.typeofDecomp.equals("WvsLR")){
			LMatrix_nTimeshk=new Matrix(tokSize,_contextSize*(k_dim));
			LTMatrix_nTimeshk=new Matrix(_contextSize*(k_dim),tokSize);
			
			RMatrix_nTimeshk=new Matrix(tokSize,_contextSize*(k_dim));
			RTMatrix_nTimeshk=new Matrix(_contextSize*(k_dim),tokSize);
		}	

		///////////////////////
		while(idx_tok<doc.size()){
			int tok=doc.get(idx_tok);
			for(int i=1;i<=_contextSize;i++){
				
			if(!_opt.typeofDecomp.equals("2viewWvsR")&& !_opt.typeofDecomp.equals("WvsR")){	
				if (idx_tok-i>=0){
					for (int j=0;j<k_dim;j++){	
						
						LMatrix_nTimeshk.set(idx_tok, (i-1)*(k_dim)+j, kdimEigDict.get(doc.get(idx_tok-i), j));
						LTMatrix_nTimeshk.set((i-1)*(k_dim)+j,idx_tok, kdimEigDict.get(doc.get(idx_tok-i), j));
					}
				}
			}
			if(!_opt.typeofDecomp.equals("2viewWvsL")&& !_opt.typeofDecomp.equals("WvsL")){	
				if (idx_tok+i <doc.size()){
						for (int j=0;j<k_dim;j++){	
							
						RMatrix_nTimeshk.set(idx_tok, (i-1)*(k_dim)+j, kdimEigDict.get(doc.get(idx_tok+i), j));
						RTMatrix_nTimeshk.set((i-1)*(k_dim)+j,idx_tok, kdimEigDict.get(doc.get(idx_tok+i), j));
					}
				}
			}
		}
			idx_tok++;
			
		}
}		

if(_opt.typeofDecomp.equals("2viewWvsL")|| _opt.typeofDecomp.equals("WvsL") || _opt.typeofDecomp.equals("TwoStepLRvsW")){		
	LTLMatrix_hkTimeshk.plusEquals(LTMatrix_nTimeshk.times(LMatrix_nTimeshk));
	
}	
	
if(_opt.typeofDecomp.equals("2viewWvsR")|| _opt.typeofDecomp.equals("WvsR") || _opt.typeofDecomp.equals("TwoStepLRvsW")){
	RTRMatrix_hkTimeshk.plusEquals(RTMatrix_nTimeshk.times(RMatrix_nTimeshk));
}
	
if(_opt.typeofDecomp.equals("TwoStepLRvsW")){
	LTLMatrix_hkTimeshk.plusEquals(LTMatrix_nTimeshk.times(LMatrix_nTimeshk));
	RTRMatrix_hkTimeshk.plusEquals(RTMatrix_nTimeshk.times(RMatrix_nTimeshk));
	
	
}

if(_opt.typeofDecomp.equals("2viewWvsLR")|| _opt.typeofDecomp.equals("WvsLR")){
	
	LRMatrix_nTimes2hk=concatenateLR(MatrixFormatConversion.createDenseMatrixCOLT(LMatrix_nTimeshk),MatrixFormatConversion.createDenseMatrixCOLT(RMatrix_nTimeshk));
	
	LRTMatrix_2hkTimesn=concatenateLRT(MatrixFormatConversion.createDenseMatrixCOLT(LTMatrix_nTimeshk),MatrixFormatConversion.createDenseMatrixCOLT(RTMatrix_nTimeshk));
	
	
	LRTLRMatrix_2hkTimes2hk.plusEquals(MatrixFormatConversion.createDenseMatrixJAMA(LRTMatrix_2hkTimesn).times(MatrixFormatConversion.createDenseMatrixJAMA(LRMatrix_nTimes2hk)));
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
		
		return MatrixFormatConversion.createDenseMatrixCOLT(LTRMatrix_hkTimeshk); 
	}
	
	public DenseDoubleMatrix2D getRTLDenseMatrix(){
		return MatrixFormatConversion.createDenseMatrixCOLT(RTLMatrix_hkTimeshk); 				
	}
	
	public DenseDoubleMatrix2D getLTLDenseMatrix(){
		return MatrixFormatConversion.createDenseMatrixCOLT(LTLMatrix_hkTimeshk);
	}
	
	public DenseDoubleMatrix2D getRTRDenseMatrix(){
		return MatrixFormatConversion.createDenseMatrixCOLT(RTRMatrix_hkTimeshk);
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
		return MatrixFormatConversion.createDenseMatrixCOLT(LRTLRMatrix_2hkTimes2hk);
	}
	
	
	/////
	
	public FlexCompRowMatrix getWnMatrix(){
		
		return WMatrix_nTimesv;
	}
	
	public FlexCompRowMatrix getLnMatrix(){
			return LMatrix_nTimeshv;
		
	}
	
	public FlexCompRowMatrix getRnMatrix(){
			return RMatrix_nTimeshv;
	}
	
	
public FlexCompRowMatrix getWnTMatrix(){
		
		return WTMatrix_nTimesv;
	}
	
	public FlexCompRowMatrix getLnTMatrix(){
			return LTMatrix_nTimeshv;
		
	}
	
	public FlexCompRowMatrix getRnTMatrix(){
			return RTMatrix_nTimeshv;
	}
	
/*	
	public FlexCompRowMatrix concatenateLR(FlexCompRowMatrix lProjectionMatrix,
			FlexCompRowMatrix rProjectionMatrix) {
		FlexCompRowMatrix finalProjection=new FlexCompRowMatrix(lProjectionMatrix.rows(),(lProjectionMatrix.columns()+rProjectionMatrix.columns()));
		
		lProjectionMatrix.forEachNonZero(arg0)
		
		
		for (int i=0;i<lProjectionMatrix.rows();i++){
			for(int j=0; j<lProjectionMatrix.columns();j++){
				finalProjection.set(i, j, lProjectionMatrix.get(i, j));
				finalProjection.set(i, j+lProjectionMatrix.columns(), rProjectionMatrix.get(i, j));
			}
		}
		return finalProjection;
	}
*/	
	
	
	
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
	
	public Matrix concatenate(Matrix lProjectionMatrix,
			int _num) {
		Matrix finalProjection=new Matrix(lProjectionMatrix.getRowDimension()*_num,lProjectionMatrix.getColumnDimension());
		
		for (int i=0;i<lProjectionMatrix.getRowDimension();i++){
			for(int j=0; j<lProjectionMatrix.getColumnDimension();j++){
				for(int k=0;k< _num;k++){
					finalProjection.set((k+1)*i, j, lProjectionMatrix.get(i, j));
				}
				
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
/*	
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
*/	
	public DenseDoubleMatrix2D concatenateLRT(DenseDoubleMatrix2D lnTMatrix, DenseDoubleMatrix2D rnTMatrix) {
		
		DenseDoubleMatrix2D finalProjection=new DenseDoubleMatrix2D((lnTMatrix.rows()+rnTMatrix.rows()),lnTMatrix.columns());
		for (int i=0;i<(lnTMatrix.rows()+rnTMatrix.rows());i++){
			for(int j=0; j<lnTMatrix.columns();j++){
				if(i <lnTMatrix.rows())
					finalProjection.set(i, j, lnTMatrix.get(i, j));
				else
					finalProjection.set(i, j, rnTMatrix.get(i-lnTMatrix.rows(), j));
			}
		}
		return finalProjection;
	}
	
	
	public Matrix concatenateLRT(Matrix lnTMatrix,
			Matrix rnTMatrix) {
		
		Matrix finalProjection=new Matrix((lnTMatrix.getRowDimension()+rnTMatrix.getRowDimension()),lnTMatrix.getColumnDimension());
		for (int i=0;i<(lnTMatrix.getRowDimension()+rnTMatrix.getRowDimension());i++){
			for(int j=0; j<lnTMatrix.getColumnDimension();j++){
				if(i <lnTMatrix.getRowDimension())
					finalProjection.set(i, j, lnTMatrix.get(i, j));
				else
					finalProjection.set(i, j, rnTMatrix.get(i-lnTMatrix.getRowDimension(), j));
			}
		}
		return finalProjection;
	}
	
	
	////////////////////
	public FlexCompRowMatrix concatenateLR(FlexCompRowMatrix lProjectionMatrix,
			FlexCompRowMatrix rProjectionMatrix) {
		FlexCompRowMatrix finalProjection=new FlexCompRowMatrix(lProjectionMatrix.numRows(),(lProjectionMatrix.numColumns()+rProjectionMatrix.numColumns()));
		
		
		Iterator<MatrixEntry> lEntry = lProjectionMatrix.iterator();
		Iterator<MatrixEntry> rEntry = rProjectionMatrix.iterator();
		double ent=0;
		
		while(lEntry.hasNext())
			{
			MatrixEntry ment = lEntry.next();
			ent =ment.get();
			finalProjection.set(ment.row(), ment.column(), ent);		
			}
		while(rEntry.hasNext())
		{
		MatrixEntry ment = rEntry.next();
		ent =ment.get();
		finalProjection.set(ment.row(), (lProjectionMatrix.numColumns()+ment.column()), ent);		
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
			System.out.println("==Created Omega Matrix==");
		return Omega;
	}
	
	public DenseDoubleMatrix2D getOmegaMatrix1Stage(int rows){//Refer Tropp's notation
		Random r= new Random();
		DenseDoubleMatrix2D Omega;
		
			Omega= new DenseDoubleMatrix2D(rows,2*_num_hidden+20);//Oversampled the rank k
			for (int i=0;i<(rows);i++){
				for (int j=0;j<2*_num_hidden+20;j++)
					Omega.set(i,j,r.nextGaussian());
			}
			System.out.println("==Created Omega Matrix==");
		return Omega;
	}
	
	public Matrix initializeRandomly(int rows){//Refer Tropp's notation
		Random r= new Random();
		Matrix Omega;
		
			Omega= new Matrix(rows,_num_hidden);//Oversampled the rank k
			for (int i=0;i<(rows);i++){
				for (int j=0;j<_num_hidden;j++)
					Omega.set(i,j,r.nextGaussian());
			}
		return Omega;
	}
	
	public Matrix initializeRandomly1Stage(int rows){//Refer Tropp's notation
		Random r= new Random();
		Matrix Omega;
		
			Omega= new Matrix(rows,2*_num_hidden);//Oversampled the rank k
			for (int i=0;i<(rows);i++){
				for (int j=0;j<2*_num_hidden;j++)
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

	public Matrix generateProjections(Matrix matrixEig, Matrix matrixContext,Matrix matrixContextR) {
		
		DenseDoubleMatrix2D tempM=new DenseDoubleMatrix2D(_opt.contextSizeOneSide,matrixEig.getColumnDimension());
		DenseDoubleMatrix2D tempM1=new DenseDoubleMatrix2D(_opt.contextSizeOneSide,matrixEig.getColumnDimension());
		tempM1=MatrixFormatConversion.createDenseMatrixCOLT(matrixEig);
		
		computeTrainLRMatrices();
		DenseDoubleMatrix2D L=new DenseDoubleMatrix2D((int) _numTok,_num_hidden);
		DenseDoubleMatrix2D R=new DenseDoubleMatrix2D((int) _numTok,_num_hidden);
		
		DenseDoubleMatrix2D LU=new DenseDoubleMatrix2D((int) _numTok,_num_hidden);
		DenseDoubleMatrix2D RU=new DenseDoubleMatrix2D((int) _numTok,_num_hidden);
		
		DenseDoubleMatrix2D contextLR=new DenseDoubleMatrix2D((int) _numTok,_num_hidden);
		
		FlexCompRowMatrix LRn2hv=new FlexCompRowMatrix((int) _numTok,2*LMatrix_nTimeshv.numColumns());
		
		DenseDoubleMatrix2D W=new DenseDoubleMatrix2D((int) _numTok,_num_hidden);
		DenseDoubleMatrix2D contextSpecificEmbed=new DenseDoubleMatrix2D((int) _numTok,2*_num_hidden);
		DenseDoubleMatrix2D contextSpecificEmbedWLR=new DenseDoubleMatrix2D((int) _numTok,3*_num_hidden);
		DenseDoubleMatrix2D contextWL=new DenseDoubleMatrix2D((int) _numTok,2*_num_hidden);
		
		
		System.out.println("Computed Train Matrices");
		
		LRn2hv=concatenateLR(LMatrix_nTimeshv, RMatrix_nTimeshv);
		System.out.println("Concatenated Train Matrices");
		
		MatrixFormatConversion.createSparseMatrixCOLT(WMatrix_nTimesv).zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixEig), W);
		System.out.println("Computed W Embeds");
		
		if(_opt.typeofDecomp.equals("2viewWvsL") || _opt.typeofDecomp.equals("WvsL") ){
			MatrixFormatConversion.createSparseMatrixCOLT(LMatrix_nTimeshv).zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixContext), L);
			contextSpecificEmbed=(DenseDoubleMatrix2D)DoubleFactory2D.dense.appendColumns(W,L);
		}
		
		if(_opt.typeofDecomp.equals("2viewWvsR") || _opt.typeofDecomp.equals("WvsR")){
			MatrixFormatConversion.createSparseMatrixCOLT(RMatrix_nTimeshv).zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixContext), R);
			contextSpecificEmbed=(DenseDoubleMatrix2D)DoubleFactory2D.dense.appendColumns(W,R);
		}
		
		if(_opt.typeofDecomp.equals("WvsLR") || _opt.typeofDecomp.equals("2viewWvsLR")|| _opt.typeofDecomp.equals("TwoStepLRvsW")|| _opt.typeofDecomp.equals("LRMVL1")){
		
			if(_opt.typeofDecomp.equals("2viewWvsLR") || _opt.typeofDecomp.equals("WvsLR")){
				MatrixFormatConversion.createSparseMatrixCOLT(LRn2hv).zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixContext), contextLR);
				contextSpecificEmbed=(DenseDoubleMatrix2D)DoubleFactory2D.dense.appendColumns(W,contextLR);
			}
			else{
				
				if(_opt.typeofDecomp.equals("LRMVL1")){
				
					if(_opt.contextSizeOneSide==1){
						MatrixFormatConversion.createSparseMatrixCOLT(LMatrix_nTimeshv).zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixEig), LU);
						MatrixFormatConversion.createSparseMatrixCOLT(RMatrix_nTimeshv).zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixEig), RU);		
					}
					else
					{
						for(int i=1; i<_opt.contextSizeOneSide;i++){
							tempM=concatenateLRT(tempM1,MatrixFormatConversion.createDenseMatrixCOLT(matrixEig));	
							tempM1=(DenseDoubleMatrix2D)tempM.copy();
						}
						MatrixFormatConversion.createSparseMatrixCOLT(LMatrix_nTimeshv).zMult(tempM, LU);
						MatrixFormatConversion.createSparseMatrixCOLT(RMatrix_nTimeshv).zMult(tempM, RU);
					}
					
					LU.zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixContext), L);
					RU.zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixContextR), R);
				
					contextWL=(DenseDoubleMatrix2D)DoubleFactory2D.dense.appendColumns(W,L);
					contextSpecificEmbedWLR=(DenseDoubleMatrix2D)DoubleFactory2D.dense.appendColumns(contextWL,R);
				}
				else{
					MatrixFormatConversion.createSparseMatrixCOLT(LMatrix_nTimeshv).zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixContext), L);
					MatrixFormatConversion.createSparseMatrixCOLT(RMatrix_nTimeshv).zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixContextR), R);
				
					contextWL=(DenseDoubleMatrix2D)DoubleFactory2D.dense.appendColumns(W,L);
					contextSpecificEmbedWLR=(DenseDoubleMatrix2D)DoubleFactory2D.dense.appendColumns(contextWL,R);
				
					
				}
			}
			
		}
		
		System.out.println("Before Return");
		if(_opt.typeofDecomp.equals("TwoStepLRvsW") || _opt.typeofDecomp.equals("LRMVL1") )
			return MatrixFormatConversion.createDenseMatrixJAMA(contextSpecificEmbedWLR);
		else
			return MatrixFormatConversion.createDenseMatrixJAMA(contextSpecificEmbed);
	}

public Matrix generateProjectionsLRMVL(Matrix matrixEig, Matrix matrixContext,Matrix matrixContextR) {
		
		computeTrainLRMatrices();
		Object[] LRViews=computeLRMVLCovMatricesTrain(matrixEig);
		Matrix L=new Matrix((int) _numTok,_num_hidden);
		Matrix R=new Matrix((int) _numTok,_num_hidden);
		
		
		DenseDoubleMatrix2D W=new DenseDoubleMatrix2D((int) _numTok,_num_hidden);
		DenseDoubleMatrix2D contextSpecificEmbedWLR=new DenseDoubleMatrix2D((int) _numTok,3*_num_hidden);
		DenseDoubleMatrix2D contextWL=new DenseDoubleMatrix2D((int) _numTok,2*_num_hidden);
		
		
		System.out.println("Computed Train Matrices");
		
		
		MatrixFormatConversion.createSparseMatrixCOLT( getWnMatrix()).zMult(MatrixFormatConversion.createDenseMatrixCOLT(matrixEig), W);
		System.out.println("Computed W Embeds");
		
		L=(Matrix)LRViews[0];
		R=(Matrix)LRViews[1];
		
		contextWL=(DenseDoubleMatrix2D)DoubleFactory2D.dense.appendColumns(W,MatrixFormatConversion.createDenseMatrixCOLT(L.times(matrixContext)));
		contextSpecificEmbedWLR=(DenseDoubleMatrix2D)DoubleFactory2D.dense.appendColumns(contextWL,MatrixFormatConversion.createDenseMatrixCOLT(R.times(matrixContextR)));
				
		
		System.out.println("Before Return");
		
		
	return MatrixFormatConversion.createDenseMatrixJAMA(contextSpecificEmbedWLR);
}

}
