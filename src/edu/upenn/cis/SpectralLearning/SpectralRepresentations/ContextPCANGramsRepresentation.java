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
import edu.umbc.cs.maple.utils.JamaUtils;
import edu.upenn.cis.SpectralLearning.Data.Corpus;
import edu.upenn.cis.SpectralLearning.Data.Document;
import edu.upenn.cis.SpectralLearning.IO.Options;
import edu.upenn.cis.SpectralLearning.IO.ReadDataFile;
import edu.upenn.cis.SpectralLearning.MathUtils.CenterScaleNormalizeUtils;
import edu.upenn.cis.SpectralLearning.MathUtils.MatrixFormatConversion;

public class ContextPCANGramsRepresentation extends SpectralRepresentation implements Serializable {

	private int _vocab_size;
	//private Corpus _corpus;
	ReadDataFile _rin;
	SparseDoubleMatrix2D CMatrix_vTimeslv,CTMatrix_vTimeslv,WMatrix_nTimesv,WMatrix_vTimesv,WLMatrix3gram,WLTMatrix3gram,
	WRMatrix3gram,WRTMatrix3gram,WLMatrix5gram,WLTMatrix5gram,WRMatrix5gram,WRTMatrix5gram;
	int _contextSize;
	static final long serialVersionUID = 42L;
	long _numTok;
	Object[] _allDocs;
	
	public ContextPCANGramsRepresentation(Options opt, long numTok, ReadDataFile rin,Object[] all_Docs) {
		super(opt, numTok);
		_vocab_size=super._opt.vocabSize;
		_rin=rin;
		_contextSize=opt.numGrams-1;
		_allDocs=all_Docs;		
	}
	
	public void computeLRContextMatrices(){
		
		HashMap<Double,Double> hMCounts=new HashMap<Double,Double>();
		CMatrix_vTimeslv=new SparseDoubleMatrix2D(_vocab_size+1,_opt.numLabels*(_vocab_size+1));
		CTMatrix_vTimeslv=new SparseDoubleMatrix2D(_opt.numLabels*(_vocab_size+1),_vocab_size+1);
		WMatrix_vTimesv=new SparseDoubleMatrix2D(_vocab_size+1,(_vocab_size+1));
		HashMap<Double, Integer> hMap=new HashMap<Double,Integer>();
		
				int idx_doc=0;
				hMap=(HashMap<Double, Integer>)_allDocs[0];
				double[] vals=new double[2];
				CenterScaleNormalizeUtils cUtils=new CenterScaleNormalizeUtils(_opt);
				
				for(Double keys: hMap.keySet()){
					vals=cUtils.cantorPairingInverseMap(keys);
					if (hMCounts.get(vals[0]) !=null)
						hMCounts.put(vals[0], hMCounts.get(vals[0])+hMap.get(keys));
					else
						hMCounts.put(vals[0], (double)hMap.get(keys));
				}	
	
			idx_doc=0;
				
				for(Double keys: hMap.keySet()){
					vals=cUtils.cantorPairingInverseMap(keys);
					CMatrix_vTimeslv.set((int)vals[0], (int)vals[1], hMap.get(keys));
					CTMatrix_vTimeslv.set((int)vals[1],(int)vals[0], hMap.get(keys));
				}	
	
				for(int i=0; i<_vocab_size+1;i++){
					WMatrix_vTimesv.set(i, i, hMCounts.get((double)i));
				}
			
		}
	
public void computeLRContextMatricesSingleVocab(){
		
		CMatrix_vTimeslv=new SparseDoubleMatrix2D(_vocab_size+1,_contextSize*(_vocab_size+1));
		CTMatrix_vTimeslv=new SparseDoubleMatrix2D(_contextSize*(_vocab_size+1),_vocab_size+1);
		
		WLMatrix3gram=new SparseDoubleMatrix2D(_vocab_size+1,(_vocab_size+1));
		WLTMatrix3gram=new SparseDoubleMatrix2D((_vocab_size+1),_vocab_size+1);
		
		WRMatrix3gram=new SparseDoubleMatrix2D(_vocab_size+1,(_vocab_size+1));
		WRTMatrix3gram=new SparseDoubleMatrix2D((_vocab_size+1),_vocab_size+1);
		
		
		WLMatrix5gram=new SparseDoubleMatrix2D(_vocab_size+1,(_contextSize/2)*(_vocab_size+1));
		WLTMatrix5gram=new SparseDoubleMatrix2D((_contextSize/2)*(_vocab_size+1),_vocab_size+1);
		
		WRMatrix5gram=new SparseDoubleMatrix2D(_vocab_size+1,(_contextSize/2)*(_vocab_size+1));
		WRTMatrix5gram=new SparseDoubleMatrix2D((_contextSize/2)*(_vocab_size+1),_vocab_size+1);
		
		WMatrix_vTimesv=new SparseDoubleMatrix2D(_vocab_size+1,(_vocab_size+1));	
		
		HashMap<Double,Double> hMCounts1=new HashMap<Double,Double>();
		HashMap<Double,Double> hMCounts2=new HashMap<Double,Double>();
		HashMap<Double,Double> hMCounts3=new HashMap<Double,Double>();
		HashMap<Double,Double> hMCounts4=new HashMap<Double,Double>();
		
		HashMap<Double, Integer> hMap1=new HashMap<Double,Integer>();
		HashMap<Double, Integer> hMap2=new HashMap<Double,Integer>();
		HashMap<Double, Integer> hMap3=new HashMap<Double,Integer>();
		HashMap<Double, Integer> hMap4=new HashMap<Double,Integer>();
		
		double[] vals=new double[2];
		CenterScaleNormalizeUtils cUtils=new CenterScaleNormalizeUtils(_opt);
				
		hMap1=(HashMap<Double, Integer>)_allDocs[0];
		hMCounts1=buildCountMaps(hMap1);
			
				
		for(Double keys: hMap1.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			CMatrix_vTimeslv.set((int)vals[0], (int)vals[1], hMap1.get(keys));
			CTMatrix_vTimeslv.set((int)vals[1],(int)vals[0], hMap1.get(keys));

		}

		for(int i=0; i<_vocab_size+1;i++){
			
			double existingCount1=WMatrix_vTimesv.get(i, i);
			try{
				WMatrix_vTimesv.set(i, i, existingCount1+ Math.ceil(hMCounts1.get((double)i)/2));
			}
			catch( Exception e ){
				WMatrix_vTimesv.set(i, i, existingCount1);//Do Nothing if all the hashmaps don't contain a given word.
			}
			}
		
		if(_opt.numGrams==3 && !_opt.typeofDecomp.equals("TwoStepLRvsW")){
			
				hMap2=(HashMap<Double, Integer>)_allDocs[1];
				hMCounts2=buildCountMaps(hMap2);
			
			for(Double keys: hMap2.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				CMatrix_vTimeslv.set((int)vals[0], (_vocab_size+1) +(int)vals[1],  hMap2.get(keys));
				CTMatrix_vTimeslv.set((_vocab_size+1)+(int)vals[1],(int)vals[0], hMap2.get(keys));
			}
			
			for(int i=0; i<_vocab_size+1;i++){
				
				double existingCount=WMatrix_vTimesv.get(i, i);
				try{
					WMatrix_vTimesv.set(i, i, existingCount+ Math.ceil(hMCounts2.get((double)i)/2));
				}
				catch( Exception e ){
					WMatrix_vTimesv.set(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
				}
				}
			//for(int i=0; i<_vocab_size+1;i++){
			//	WMatrix_vTimesv.set(i, i, WMatrix_vTimesv.get(i, i)/2);
			//}
			
		}
		/////////
		if(_opt.numGrams==3 && _opt.typeofDecomp.equals("TwoStepLRvsW")){
			
			hMap2=(HashMap<Double, Integer>)_allDocs[1];
			hMCounts2=buildCountMaps(hMap2);
		
		for(Double keys: hMap1.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			WLMatrix3gram.set((int)vals[0],(int)vals[1],  hMap1.get(keys));
			WLTMatrix3gram.set((int)vals[1],(int)vals[0], hMap1.get(keys));
		}	
			
		for(Double keys: hMap2.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			WRMatrix3gram.set((int)vals[0],(int)vals[1],  hMap2.get(keys));
			WRTMatrix3gram.set((int)vals[1],(int)vals[0], hMap2.get(keys));
		}
		
		for(int i=0; i<_vocab_size+1;i++){
			
			double existingCount=WMatrix_vTimesv.get(i, i);
			try{
				WMatrix_vTimesv.set(i, i, existingCount+ Math.ceil(hMCounts2.get((double)i)/2));
			}
			catch( Exception e ){
				WMatrix_vTimesv.set(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
			}
			}
		//for(int i=0; i<_vocab_size+1;i++){
		//	WMatrix_vTimesv.set(i, i, WMatrix_vTimesv.get(i, i)/2);
		//}
		
	}
		
		
		
		
		if(_opt.numGrams==5 && !_opt.typeofDecomp.equals("TwoStepLRvsW")){
			
				hMap2=(HashMap<Double, Integer>)_allDocs[1];
				hMCounts2=buildCountMaps(hMap2);
				
				hMap3=(HashMap<Double, Integer>)_allDocs[2];
				hMCounts3=buildCountMaps(hMap3);
				
				hMap4=(HashMap<Double, Integer>)_allDocs[3];
				hMCounts4=buildCountMaps(hMap4);	
		
			for(Double keys: hMap2.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				CMatrix_vTimeslv.set((int)vals[0], (_vocab_size+1) +(int)vals[1],  hMap2.get(keys));
				CTMatrix_vTimeslv.set((_vocab_size+1)+(int)vals[1],(int)vals[0], hMap2.get(keys));
			}
			
			for(Double keys: hMap3.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				CMatrix_vTimeslv.set((int)vals[0], (2*(_vocab_size+1)) +(int)vals[1],  hMap3.get(keys));
				CTMatrix_vTimeslv.set((2*(_vocab_size+1))+(int)vals[1],(int)vals[0], hMap3.get(keys));
			}
			
			for(Double keys: hMap4.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				CMatrix_vTimeslv.set((int)vals[0], (3*(_vocab_size+1)) +(int)vals[1],  hMap4.get(keys));
				CTMatrix_vTimeslv.set((3*(_vocab_size+1))+(int)vals[1],(int)vals[0], hMap4.get(keys));
			}
			
			for(int i=0; i<_vocab_size+1;i++){
				
				double existingCount=WMatrix_vTimesv.get(i, i);
				try{
					WMatrix_vTimesv.set(i, i, existingCount+ Math.ceil(hMCounts2.get((double)i)/2));
				}
				catch( Exception e ){
					WMatrix_vTimesv.set(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
				}
			}
			for(int i=0; i<_vocab_size+1;i++){
				
				double existingCount=WMatrix_vTimesv.get(i, i);
				try{
					WMatrix_vTimesv.set(i, i, existingCount+ Math.ceil(hMCounts3.get((double)i)/2));
				}
				catch( Exception e ){
					WMatrix_vTimesv.set(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
				}
			}
			for(int i=0; i<_vocab_size+1;i++){
	
				double existingCount=WMatrix_vTimesv.get(i, i);
				try{
					WMatrix_vTimesv.set(i, i, existingCount+ Math.ceil(hMCounts4.get((double)i)/2));
					}
				catch( Exception e ){
					WMatrix_vTimesv.set(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
					}
			}

			//for(int i=0; i<_vocab_size+1;i++){
			//	WMatrix_vTimesv.set(i, i, WMatrix_vTimesv.get(i, i)/2);
			//}
		}
		/////////////
		if(_opt.numGrams==5 && _opt.typeofDecomp.equals("TwoStepLRvsW")){
			
			hMap2=(HashMap<Double, Integer>)_allDocs[1];
			hMCounts2=buildCountMaps(hMap2);
			
			hMap3=(HashMap<Double, Integer>)_allDocs[2];
			hMCounts3=buildCountMaps(hMap3);
			
			hMap4=(HashMap<Double, Integer>)_allDocs[3];
			hMCounts4=buildCountMaps(hMap4);
			
			
		for(Double keys: hMap1.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			WLMatrix5gram.set((int)vals[0], (int)vals[1], hMap1.get(keys));
			WLTMatrix5gram.set((int)vals[1],(int)vals[0], hMap1.get(keys));
		}	
	
		for(Double keys: hMap2.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			WLMatrix5gram.set((int)vals[0], (_vocab_size+1) +(int)vals[1],  hMap2.get(keys));
			WLTMatrix5gram.set((_vocab_size+1)+(int)vals[1],(int)vals[0], hMap2.get(keys));
		}
		
		for(Double keys: hMap3.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			WRMatrix5gram.set((int)vals[0], (int)vals[1],  hMap3.get(keys));
			WRTMatrix5gram.set((int)vals[1],(int)vals[0], hMap3.get(keys));
		}
		
		for(Double keys: hMap4.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			WRMatrix5gram.set((int)vals[0], (_vocab_size+1) +(int)vals[1],  hMap4.get(keys));
			WRTMatrix5gram.set((_vocab_size+1)+(int)vals[1],(int)vals[0], hMap4.get(keys));
		}
		
		for(int i=0; i<_vocab_size+1;i++){
			
			double existingCount=WMatrix_vTimesv.get(i, i);
			try{
				WMatrix_vTimesv.set(i, i, existingCount+Math.ceil(hMCounts2.get((double)i)/2));
			}
			catch( Exception e ){
				WMatrix_vTimesv.set(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
			}
		}
		for(int i=0; i<_vocab_size+1;i++){
			
			double existingCount=WMatrix_vTimesv.get(i, i);
			try{
				WMatrix_vTimesv.set(i, i, existingCount+ Math.ceil(hMCounts3.get((double)i)/2));
			}
			catch( Exception e ){
				WMatrix_vTimesv.set(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
			}
		}
		for(int i=0; i<_vocab_size+1;i++){

			double existingCount=WMatrix_vTimesv.get(i, i);
			try{
				WMatrix_vTimesv.set(i, i, existingCount+ Math.ceil(hMCounts4.get((double)i)/2));
				}
			catch( Exception e ){
				WMatrix_vTimesv.set(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
				}
		}

		//for(int i=0; i<_vocab_size+1;i++){
		//	WMatrix_vTimesv.set(i, i, WMatrix_vTimesv.get(i, i)/2);
		//}
	}
}
		
		
		
	
	


HashMap<Double,Double> buildCountMaps(HashMap<Double,Integer> hMap){
	HashMap<Double,Double> hMCounts=new HashMap<Double,Double>();
	double[] vals=new double[2];
	CenterScaleNormalizeUtils cUtils=new CenterScaleNormalizeUtils(_opt);
	
	
	for(Double keys: hMap.keySet()){
		vals=cUtils.cantorPairingInverseMap(keys);
		if (hMCounts.get(vals[0]) !=null)
			hMCounts.put(vals[0], hMCounts.get(vals[0])+hMap.get(keys));
		else
			hMCounts.put(vals[0], (double)hMap.get(keys));
		if (hMCounts.get(vals[1]) !=null)
			hMCounts.put(vals[1], hMCounts.get(vals[1])+hMap.get(keys));
		else
			hMCounts.put(vals[1], (double)hMap.get(keys));
}
	return hMCounts;
}

	
	public SparseDoubleMatrix2D getContextMatrix(){
			return CMatrix_vTimeslv;
	}
	
	public SparseDoubleMatrix2D getContextMatrixT(){
		return CTMatrix_vTimeslv;
}
	
	
	public SparseDoubleMatrix2D getWL3gramMatrix(){
		return WLMatrix3gram;
	}

	public SparseDoubleMatrix2D getWLT3gramMatrix(){
		return WLTMatrix3gram;
	}
	
	public SparseDoubleMatrix2D getWR3gramMatrix(){
		return WRMatrix3gram;
	}

	public SparseDoubleMatrix2D getWRT3gramMatrix(){
		return WRTMatrix3gram;
	}
	public SparseDoubleMatrix2D getWL5gramMatrix(){
		return WLMatrix5gram;
	}

	public SparseDoubleMatrix2D getWLT5gramMatrix(){
		return WLTMatrix5gram;
	}
	
	public SparseDoubleMatrix2D getWR5gramMatrix(){
		return WRMatrix5gram;
	}

	public SparseDoubleMatrix2D getWRT5gramMatrix(){
		return WRTMatrix5gram;
	}
	

	public SparseDoubleMatrix2D getWTWMatrix(){
		return WMatrix_vTimesv;
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
	
	
	public DenseDoubleMatrix2D getOmegaMatrix(){//Refer Tropp's notation
		Random r= new Random();
		DenseDoubleMatrix2D Omega;
		Omega= new DenseDoubleMatrix2D(_opt.numLabels*(_vocab_size+1),_num_hidden+20);//Oversampled the rank k
			for (int i=0;i<_opt.numLabels*(_vocab_size+1);i++){
				for (int j=0;j<_num_hidden+20;j++)
					Omega.set(i,j,r.nextGaussian());
			}
		
		return Omega;
	}
	
	
	public void serializeContextPCANGramsRepresentation() {
		File f= new File(_opt.serializeRep+"NGrams");
		
		try{
			ObjectOutput cpcaRep=new ObjectOutputStream(new FileOutputStream(f));
			cpcaRep.writeObject(this);
			
			System.out.println("=======Serialized the ContextPCA NGrams Representation=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
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
	
	/*
	public Matrix getContextOblEmbeddings(Matrix eigenFeatDict) {
		Matrix WProjectionMatrix;
		
		WProjectionMatrix=generateWProjections(_allDocs,_rin.getSortedWordList(),eigenFeatDict);
		
		
		return WProjectionMatrix;
	}
	*/

	
}
