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
import java.util.HashMap;
import java.util.Random;

import jeigen.SparseMatrixLil;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import edu.upenn.cis.swell.IO.Options;
import edu.upenn.cis.swell.IO.ReadDataFile;
import edu.upenn.cis.swell.MathUtils.CenterScaleNormalizeUtils;
import edu.upenn.cis.swell.MathUtils.MatrixFormatConversion;

public class ContextPCANGramsRepresentation extends SpectralRepresentation implements Serializable {

	private int _vocab_size;
	//private Corpus _corpus;
	ReadDataFile _rin;
	FlexCompRowMatrix CMatrix_vTimeslv,CTMatrix_vTimeslv,WMatrix_nTimesv,WMatrix_vTimesv,WLMatrix3gram,WLTMatrix3gram,
	WRMatrix3gram,WRTMatrix3gram,WLMatrix5gram,WLTMatrix5gram,WRMatrix5gram,WRTMatrix5gram,WL_OR_WRMatrix3gram,WLT_OR_WRTMatrix3gram,
	WL_OR_WRMatrix5gram,WLT_OR_WRTMatrix5gram,L1L2_OR_R1R2_L1R1Matrix_vTimesv,L1L3_OR_R1R3_L1R2Matrix_vTimesv,L1L4_OR_R1R4_L2R1Matrix_vTimesv
	,L2L3_OR_R2R3_L2R2Matrix_vTimesv,L2L4_OR_R2R4_L1L2Matrix_vTimesv,L3L4_OR_R3R4_R1R2Matrix_vTimesv,L1L2_OR_R1R2_L1R1Matrix_vTimesvT,L1L3_OR_R1R3_L1R2Matrix_vTimesvT,L1L4_OR_R1R4_L2R1Matrix_vTimesvT
	,L2L3_OR_R2R3_L2R2Matrix_vTimesvT,L2L4_OR_R2R4_L1L2Matrix_vTimesvT,L3L4_OR_R3R4_R1R2Matrix_vTimesvT;
	
	
	
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
		CMatrix_vTimeslv=new FlexCompRowMatrix(_vocab_size+1,_opt.numLabels*(_vocab_size+1));
		CTMatrix_vTimeslv=new FlexCompRowMatrix(_opt.numLabels*(_vocab_size+1),_vocab_size+1);
		WMatrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
		HashMap<Double, Double> hMap=new HashMap<Double,Double>();
		
				int idx_doc=0;
				hMap=(HashMap<Double, Double>)_allDocs[0];
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
					CMatrix_vTimeslv.add((int)vals[0], (int)vals[1], hMap.get(keys));
					CTMatrix_vTimeslv.add((int)vals[1],(int)vals[0], hMap.get(keys));
				}	
	
				for(int i=0; i<_vocab_size+1;i++){
					WMatrix_vTimesv.add(i, i, hMCounts.get((double)i));
				}
			
		}
	
public void computeLRContextMatricesSingleVocab(){
		
		CMatrix_vTimeslv=new FlexCompRowMatrix(_vocab_size+1,_contextSize*(_vocab_size+1));
		CTMatrix_vTimeslv=new FlexCompRowMatrix(_contextSize*(_vocab_size+1),(_vocab_size+1));
		
		WMatrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
		
		HashMap<Double,Double> hMCounts1=new HashMap<Double,Double>();
		HashMap<Double,Double> hMap1=new HashMap<Double,Double>();
		HashMap<Double,Double> hMCounts2,hMCounts3,hMCounts4,hMap2,hMap3,hMap4, hMap5, hMap6, hMap7, hMap8, hMap9, hMap10;
		
		
		
		double[] vals=new double[2];
		CenterScaleNormalizeUtils cUtils=new CenterScaleNormalizeUtils(_opt);
				
		hMap1=(HashMap<Double, Double>)_allDocs[0];
		hMCounts1=buildCountMaps(hMap1);
			
		System.out.println("+++++Entering: Populate the context matrix+++++");
				
		for(Double keys: hMap1.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			CMatrix_vTimeslv.add((int)vals[0], (int)vals[1], hMap1.get(keys));
			CTMatrix_vTimeslv.add((int)vals[1],(int)vals[0], hMap1.get(keys));

		}
		
		
		System.out.println("+++++Populated the context matrix+++++");
		
		for(int i=0; i<_vocab_size+1;i++){
			
			double existingCount1=WMatrix_vTimesv.get(i, i);
			try{
				WMatrix_vTimesv.add(i, i, existingCount1+ Math.ceil(hMCounts1.get((double)i)/2));
			}
			catch( Exception e ){
				WMatrix_vTimesv.add(i, i, existingCount1);//Do Nothing if all the hashmaps don't contain a given word.
			}
			}
		System.out.println("+++++Populated the word matrix+++++");
		
		if(_opt.numGrams==2){
			L1L2_OR_R1R2_L1R1Matrix_vTimesv=WMatrix_vTimesv;
			L1L2_OR_R1R2_L1R1Matrix_vTimesvT=WMatrix_vTimesv;
		}
		
		
		
		
		if(_opt.numGrams==3 && !_opt.typeofDecomp.equals("TwoStepLRvsW")){
			
			
						
			//Used for 3 as well as 5 grams
			L1L2_OR_R1R2_L1R1Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L1L2_OR_R1R2_L1R1Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			 hMCounts2=new HashMap<Double,Double>();
			 hMap2=new HashMap<Double,Double>();
			 hMap3=new HashMap<Double,Double>();
			

			
				hMap2=(HashMap<Double, Double>)_allDocs[1];
				hMap3=(HashMap<Double, Double>)_allDocs[2];
				
				hMCounts2=buildCountMaps(hMap2);
				
			
			for(Double keys: hMap2.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				CMatrix_vTimeslv.add((int)vals[0], (_vocab_size+1) +(int)vals[1],  hMap2.get(keys));
				CTMatrix_vTimeslv.add((_vocab_size+1)+(int)vals[1],(int)vals[0], hMap2.get(keys));
			}
			
			for(Double keys: hMap3.keySet()){
				
				vals=cUtils.cantorPairingInverseMap(keys);
				L1L2_OR_R1R2_L1R1Matrix_vTimesv.add((int)vals[0], (int)vals[1], hMap3.get(keys));
				L1L2_OR_R1R2_L1R1Matrix_vTimesvT.add((int)vals[1], (int)vals[0], hMap3.get(keys));

				}
			
			
			for(int i=0; i<_vocab_size+1;i++){
				
				double existingCount=WMatrix_vTimesv.get(i, i);
				try{
					WMatrix_vTimesv.add(i, i, existingCount+ Math.ceil(hMCounts2.get((double)i)/2));
				}
				catch( Exception e ){
					WMatrix_vTimesv.add(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
				}
				}
			//for(int i=0; i<_vocab_size+1;i++){
			//	WMatrix_vTimesv.set(i, i, WMatrix_vTimesv.get(i, i)/2);
			//}
			
		}
		/////////
		if(_opt.numGrams==3 && _opt.typeofDecomp.equals("TwoStepLRvsW")){
			
			WL_OR_WRMatrix3gram=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			WLT_OR_WRTMatrix3gram=new FlexCompRowMatrix((_vocab_size+1),_vocab_size+1);
			
			WLMatrix3gram=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			WLTMatrix3gram=new FlexCompRowMatrix((_vocab_size+1),_vocab_size+1);

			WRMatrix3gram=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			WRTMatrix3gram=new FlexCompRowMatrix((_vocab_size+1),_vocab_size+1);
			
			//Used for 3 as well as 5 grams
			L1L2_OR_R1R2_L1R1Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L1L2_OR_R1R2_L1R1Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			 hMCounts2=new HashMap<Double,Double>();
			 hMap2=new HashMap<Double,Double>();
			 hMap3=new HashMap<Double,Double>();
			
			
			hMap2=(HashMap<Double, Double>)_allDocs[1];
			hMap3=(HashMap<Double, Double>)_allDocs[2];
			hMCounts2=buildCountMaps(hMap2);
		
		for(Double keys: hMap1.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			WLMatrix3gram.add((int)vals[0],(int)vals[1],  hMap1.get(keys));
			WLTMatrix3gram.add((int)vals[1],(int)vals[0], hMap1.get(keys));
		}	
			
		for(Double keys: hMap2.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			WRMatrix3gram.add((int)vals[0],(int)vals[1],  hMap2.get(keys));
			WRTMatrix3gram.add((int)vals[1],(int)vals[0], hMap2.get(keys));
		}
		
		for(Double keys: hMap3.keySet()){
			
			vals=cUtils.cantorPairingInverseMap(keys);
			L1L2_OR_R1R2_L1R1Matrix_vTimesv.add((int)vals[0], (int)vals[1], hMap3.get(keys));
			L1L2_OR_R1R2_L1R1Matrix_vTimesvT.add((int)vals[1], (int)vals[0], hMap3.get(keys));

			}
		
		for(int i=0; i<_vocab_size+1;i++){
			
			double existingCount=WMatrix_vTimesv.get(i, i);
			try{
				WMatrix_vTimesv.add(i, i, existingCount+ Math.ceil(hMCounts2.get((double)i)/2));
			}
			catch( Exception e ){
				WMatrix_vTimesv.add(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
			}
			}
		//for(int i=0; i<_vocab_size+1;i++){
		//	WMatrix_vTimesv.set(i, i, WMatrix_vTimesv.get(i, i)/2);
		//}
		
	}
		
		
		
		
		if(_opt.numGrams==5 && !_opt.typeofDecomp.equals("TwoStepLRvsW")){
			
			//Used for 3 as well as 5 grams
			L1L2_OR_R1R2_L1R1Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L1L2_OR_R1R2_L1R1Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			
			
			L1L3_OR_R1R3_L1R2Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L1L4_OR_R1R4_L2R1Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L2L3_OR_R2R3_L2R2Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L2L4_OR_R2R4_L1L2Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L3L4_OR_R3R4_R1R2Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			
			L1L3_OR_R1R3_L1R2Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L1L4_OR_R1R4_L2R1Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L2L3_OR_R2R3_L2R2Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L2L4_OR_R2R4_L1L2Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L3L4_OR_R3R4_R1R2Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
					
			 hMCounts3=new HashMap<Double,Double>();
			 hMCounts4=new HashMap<Double,Double>();
			
			hMap4=new HashMap<Double,Double>();
			
			hMap5=new HashMap<Double,Double>();
			hMap6=new HashMap<Double,Double>();
			hMap7=new HashMap<Double,Double>();
			hMap8=new HashMap<Double,Double>();
			hMap9=new HashMap<Double,Double>();
			hMap10=new HashMap<Double,Double>();

			
			
				hMap2=(HashMap<Double, Double>)_allDocs[1];
				hMCounts2=buildCountMaps(hMap2);
				
				hMap3=(HashMap<Double, Double>)_allDocs[2];
				hMCounts3=buildCountMaps(hMap3);
				
				hMap4=(HashMap<Double, Double>)_allDocs[3];
				hMCounts4=buildCountMaps(hMap4);
				
				hMap5=(HashMap<Double, Double>)_allDocs[4];
				
				hMap6=(HashMap<Double, Double>)_allDocs[5];
				
				hMap7=(HashMap<Double, Double>)_allDocs[6];
				
				hMap8=(HashMap<Double, Double>)_allDocs[7];
				
				hMap9=(HashMap<Double, Double>)_allDocs[8];
				
				hMap10=(HashMap<Double, Double>)_allDocs[9];
				
				
			for(Double keys: hMap2.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				CMatrix_vTimeslv.add((int)vals[0], (_vocab_size+1) +(int)vals[1],  hMap2.get(keys));
				CTMatrix_vTimeslv.add((_vocab_size+1)+(int)vals[1],(int)vals[0], hMap2.get(keys));
			}
			
			for(Double keys: hMap3.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				CMatrix_vTimeslv.add((int)vals[0], (2*(_vocab_size+1)) +(int)vals[1],  hMap3.get(keys));
				CTMatrix_vTimeslv.add((2*(_vocab_size+1))+(int)vals[1],(int)vals[0], hMap3.get(keys));
			}
			
			for(Double keys: hMap4.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				CMatrix_vTimeslv.add((int)vals[0], (3*(_vocab_size+1)) +(int)vals[1],  hMap4.get(keys));
				CTMatrix_vTimeslv.add((3*(_vocab_size+1))+(int)vals[1],(int)vals[0], hMap4.get(keys));
			}
			
			
			///
			for(Double keys: hMap5.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				L1L2_OR_R1R2_L1R1Matrix_vTimesv.add((int)vals[0], (int)vals[1], hMap5.get(keys));
				L1L2_OR_R1R2_L1R1Matrix_vTimesvT.add((int)vals[1], (int)vals[0], hMap5.get(keys));
			}
				
			for(Double keys: hMap6.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				L1L3_OR_R1R3_L1R2Matrix_vTimesv.add((int)vals[0], (int)vals[1], hMap6.get(keys));
				L1L3_OR_R1R3_L1R2Matrix_vTimesvT.add((int)vals[1], (int)vals[0], hMap6.get(keys));
			}
			for(Double keys: hMap7.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				L1L4_OR_R1R4_L2R1Matrix_vTimesv.add((int)vals[0], (int)vals[1], hMap7.get(keys));
				L1L4_OR_R1R4_L2R1Matrix_vTimesvT.add((int)vals[1], (int)vals[0], hMap7.get(keys));
			}
			for(Double keys: hMap8.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				L2L3_OR_R2R3_L2R2Matrix_vTimesv.add((int)vals[0], (int)vals[1], hMap8.get(keys));
				L2L3_OR_R2R3_L2R2Matrix_vTimesvT.add((int)vals[1], (int)vals[0], hMap8.get(keys));
			}
			for(Double keys: hMap9.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				L2L4_OR_R2R4_L1L2Matrix_vTimesv.add((int)vals[0], (int)vals[1], hMap9.get(keys));
				L2L4_OR_R2R4_L1L2Matrix_vTimesvT.add((int)vals[1], (int)vals[0], hMap9.get(keys));
			}
			for(Double keys: hMap10.keySet()){
				vals=cUtils.cantorPairingInverseMap(keys);
				L3L4_OR_R3R4_R1R2Matrix_vTimesv.add((int)vals[0], (int)vals[1], hMap10.get(keys));
				L3L4_OR_R3R4_R1R2Matrix_vTimesvT.add((int)vals[1], (int)vals[0], hMap10.get(keys));
			}
			
			
			
			
			for(int i=0; i<_vocab_size+1;i++){
				
				double existingCount=WMatrix_vTimesv.get(i, i);
				try{
					WMatrix_vTimesv.add(i, i, existingCount+ Math.ceil(hMCounts2.get((double)i)/2));
				}
				catch( Exception e ){
					WMatrix_vTimesv.add(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
				}
			}
			for(int i=0; i<_vocab_size+1;i++){
				
				double existingCount=WMatrix_vTimesv.get(i, i);
				try{
					WMatrix_vTimesv.add(i, i, existingCount+ Math.ceil(hMCounts3.get((double)i)/2));
				}
				catch( Exception e ){
					WMatrix_vTimesv.add(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
				}
			}
			for(int i=0; i<_vocab_size+1;i++){
	
				double existingCount=WMatrix_vTimesv.get(i, i);
				try{
					WMatrix_vTimesv.add(i, i, existingCount+ Math.ceil(hMCounts4.get((double)i)/2));
					}
				catch( Exception e ){
					WMatrix_vTimesv.add(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
					}
			}

			//for(int i=0; i<_vocab_size+1;i++){
			//	WMatrix_vTimesv.set(i, i, WMatrix_vTimesv.get(i, i)/2);
			//}
		}
		/////////////
		if(_opt.numGrams==5 && _opt.typeofDecomp.equals("TwoStepLRvsW")){
			
			//Used for 3 as well as 5 grams
			L1L2_OR_R1R2_L1R1Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L1L2_OR_R1R2_L1R1Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			
			
			WLMatrix5gram=new FlexCompRowMatrix(_vocab_size+1,(_contextSize/2)*(_vocab_size+1));
			WLTMatrix5gram=new FlexCompRowMatrix((_contextSize/2)*(_vocab_size+1),_vocab_size+1);

			WRMatrix5gram=new FlexCompRowMatrix(_vocab_size+1,(_contextSize/2)*(_vocab_size+1));
			WRTMatrix5gram=new FlexCompRowMatrix((_contextSize/2)*(_vocab_size+1),_vocab_size+1);
			
			WL_OR_WRMatrix5gram=new FlexCompRowMatrix(_vocab_size+1,(_contextSize/2)*(_vocab_size+1));
			WLT_OR_WRTMatrix5gram=new FlexCompRowMatrix((_contextSize/2)*(_vocab_size+1),_vocab_size+1);
					
			
			L1L3_OR_R1R3_L1R2Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L1L4_OR_R1R4_L2R1Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L2L3_OR_R2R3_L2R2Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L2L4_OR_R2R4_L1L2Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L3L4_OR_R3R4_R1R2Matrix_vTimesv=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			
			L1L3_OR_R1R3_L1R2Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L1L4_OR_R1R4_L2R1Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L2L3_OR_R2R3_L2R2Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L2L4_OR_R2R4_L1L2Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
			L3L4_OR_R3R4_R1R2Matrix_vTimesvT=new FlexCompRowMatrix(_vocab_size+1,(_vocab_size+1));
					
			 hMCounts3=new HashMap<Double,Double>();
			 hMCounts4=new HashMap<Double,Double>();
			
			hMap4=new HashMap<Double,Double>();
			
			hMap5=new HashMap<Double,Double>();
			hMap6=new HashMap<Double,Double>();
			hMap7=new HashMap<Double,Double>();
			hMap8=new HashMap<Double,Double>();
			hMap9=new HashMap<Double,Double>();
			hMap10=new HashMap<Double,Double>();

			
			hMap2=(HashMap<Double, Double>)_allDocs[1];
			hMCounts2=buildCountMaps(hMap2);
			
			hMap3=(HashMap<Double, Double>)_allDocs[2];
			hMCounts3=buildCountMaps(hMap3);
			
			hMap4=(HashMap<Double, Double>)_allDocs[3];
			hMCounts4=buildCountMaps(hMap4);
			
			
		for(Double keys: hMap1.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			WLMatrix5gram.add((int)vals[0], (int)vals[1], hMap1.get(keys));
			WLTMatrix5gram.add((int)vals[1],(int)vals[0], hMap1.get(keys));
		}	
	
		for(Double keys: hMap2.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			WLMatrix5gram.add((int)vals[0], (_vocab_size+1) +(int)vals[1],  hMap2.get(keys));
			WLTMatrix5gram.add((_vocab_size+1)+(int)vals[1],(int)vals[0], hMap2.get(keys));
		}
		
		for(Double keys: hMap3.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			WRMatrix5gram.add((int)vals[0], (int)vals[1],  hMap3.get(keys));
			WRTMatrix5gram.add((int)vals[1],(int)vals[0], hMap3.get(keys));
		}
		
		for(Double keys: hMap4.keySet()){
			vals=cUtils.cantorPairingInverseMap(keys);
			WRMatrix5gram.add((int)vals[0], (_vocab_size+1) +(int)vals[1],  hMap4.get(keys));
			WRTMatrix5gram.add((_vocab_size+1)+(int)vals[1],(int)vals[0], hMap4.get(keys));
		}
		
		for(int i=0; i<_vocab_size+1;i++){
			
			double existingCount=WMatrix_vTimesv.get(i, i);
			try{
				WMatrix_vTimesv.add(i, i, existingCount+Math.ceil(hMCounts2.get((double)i)/2));
			}
			catch( Exception e ){
				WMatrix_vTimesv.add(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
			}
		}
		for(int i=0; i<_vocab_size+1;i++){
			
			double existingCount=WMatrix_vTimesv.get(i, i);
			try{
				WMatrix_vTimesv.add(i, i, existingCount+ Math.ceil(hMCounts3.get((double)i)/2));
			}
			catch( Exception e ){
				WMatrix_vTimesv.add(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
			}
		}
		for(int i=0; i<_vocab_size+1;i++){

			double existingCount=WMatrix_vTimesv.get(i, i);
			try{
				WMatrix_vTimesv.add(i, i, existingCount+ Math.ceil(hMCounts4.get((double)i)/2));
				}
			catch( Exception e ){
				WMatrix_vTimesv.add(i, i, existingCount);//Do Nothing if all the hashmaps don't contain a given word.
				}
		}

		//for(int i=0; i<_vocab_size+1;i++){
		//	WMatrix_vTimesv.set(i, i, WMatrix_vTimesv.get(i, i)/2);
		//}
	}
}
		
		
		
	
	


HashMap<Double,Double> buildCountMaps(HashMap<Double,Double> hMap){
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





///////////////
public FlexCompRowMatrix getL1L3_OR_R1R3_L1R2Matrix_vTimesv(){
	return L1L3_OR_R1R3_L1R2Matrix_vTimesv;
}

public FlexCompRowMatrix getL1L3_OR_R1R3_L1R2Matrix_vTimesvT(){
return L1L3_OR_R1R3_L1R2Matrix_vTimesvT;
}

public FlexCompRowMatrix getL1L4_OR_R1R4_L2R1Matrix_vTimesv(){
	return L1L4_OR_R1R4_L2R1Matrix_vTimesv;
}

public FlexCompRowMatrix getL1L4_OR_R1R4_L2R1Matrix_vTimesvT(){
return L1L4_OR_R1R4_L2R1Matrix_vTimesvT;
}

public FlexCompRowMatrix getL2L3_OR_R2R3_L2R2Matrix_vTimesv(){
	return L2L3_OR_R2R3_L2R2Matrix_vTimesv;
}

public FlexCompRowMatrix getL2L3_OR_R2R3_L2R2Matrix_vTimesvT(){
return L2L3_OR_R2R3_L2R2Matrix_vTimesvT;
}

public FlexCompRowMatrix getL2L4_OR_R2R4_L1L2Matrix_vTimesv(){
	return L2L4_OR_R2R4_L1L2Matrix_vTimesv;
}

public FlexCompRowMatrix getL2L4_OR_R2R4_L1L2Matrix_vTimesvT(){
return L2L4_OR_R2R4_L1L2Matrix_vTimesvT;
}

public FlexCompRowMatrix getL3L4_OR_R3R4_R1R2Matrix_vTimesv(){
	return L3L4_OR_R3R4_R1R2Matrix_vTimesv;
}

public FlexCompRowMatrix getL3L4_OR_R3R4_R1R2Matrix_vTimesvT(){
return L3L4_OR_R3R4_R1R2Matrix_vTimesvT;
}



//////////////

	public FlexCompRowMatrix getL1L2_OR_R1R2_L1R1Matrix_vTimesv(){
			return L1L2_OR_R1R2_L1R1Matrix_vTimesv;
	}
	
	public FlexCompRowMatrix getL1L2_OR_R1R2_L1R1Matrix_vTimesvT(){
		return L1L2_OR_R1R2_L1R1Matrix_vTimesvT;
}

	
	public FlexCompRowMatrix getContextMatrix(){
		return CMatrix_vTimeslv;
}
	
	
	public FlexCompRowMatrix getContextMatrixT(){
		return CTMatrix_vTimeslv;
}
	
	
	public FlexCompRowMatrix getWL3gramMatrix(){
		return WLMatrix3gram;
	}

	public FlexCompRowMatrix getWLT3gramMatrix(){
		return WLTMatrix3gram;
	}
	
	public FlexCompRowMatrix getWR3gramMatrix(){
		return WRMatrix3gram;
	}

	public FlexCompRowMatrix getWRT3gramMatrix(){
		return WRTMatrix3gram;
	}
	public FlexCompRowMatrix getWL5gramMatrix(){
		return WLMatrix5gram;
	}

	public FlexCompRowMatrix getWLT5gramMatrix(){
		return WLTMatrix5gram;
	}
	
	public FlexCompRowMatrix getWR5gramMatrix(){
		return WRMatrix5gram;
	}

	public FlexCompRowMatrix getWRT5gramMatrix(){
		return WRTMatrix5gram;
	}
	

	public FlexCompRowMatrix getWTWMatrix(){
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
	
	public FlexCompRowMatrix concatenateLR(FlexCompRowMatrix lProjectionMatrix,
			FlexCompRowMatrix rProjectionMatrix) {
		//FlexCompRowMatrix finalProjection=new FlexCompRowMatrix(lProjectionMatrix.numRows(),(lProjectionMatrix.numColumns()+rProjectionMatrix.numColumns()));
		
		SparseMatrixLil concatMatrix=new SparseMatrixLil(lProjectionMatrix.numRows(),(lProjectionMatrix.numColumns()+rProjectionMatrix.numColumns()));	
		SparseMatrixLil l=new SparseMatrixLil(lProjectionMatrix.numRows(),lProjectionMatrix.numColumns());
		SparseMatrixLil r=new SparseMatrixLil(rProjectionMatrix.numRows(),rProjectionMatrix.numColumns());
		
		System.out.println("+++Before Multiply+++");
		l=MatrixFormatConversion.createJeigenMatrix(lProjectionMatrix);
		r=MatrixFormatConversion.createJeigenMatrix(rProjectionMatrix);
		
		concatMatrix=l.concatRight(r);
		System.out.println("+++After Multiply+++");
		
		return MatrixFormatConversion.createSparseMatrixMTJFromJeigen(concatMatrix);
		
		
	}
	
	public FlexCompRowMatrix concatenateLRT(FlexCompRowMatrix lnTMatrix,
			FlexCompRowMatrix rnTMatrix) {
		
		SparseMatrixLil concatMatrix=new SparseMatrixLil((lnTMatrix.numRows()+rnTMatrix.numRows()),lnTMatrix.numColumns());	
		SparseMatrixLil lt=new SparseMatrixLil(lnTMatrix.numRows(),lnTMatrix.numColumns());
		SparseMatrixLil rt=new SparseMatrixLil(rnTMatrix.numRows(),rnTMatrix.numColumns());
		
		System.out.println("+++Before Multiply+++");
		lt=MatrixFormatConversion.createJeigenMatrix(lnTMatrix);
		rt=MatrixFormatConversion.createJeigenMatrix(rnTMatrix);
		
		concatMatrix=lt.concatDown(rt);
		System.out.println("+++After Multiply+++");
		
		return MatrixFormatConversion.createSparseMatrixMTJFromJeigen(concatMatrix);
		
		
	}
	
	

	public SparseDoubleMatrix2D  concatenateMultiRow(SparseDoubleMatrix2D xtx,
			SparseDoubleMatrix2D l1l2_OR_R1R2_L1R1,
			SparseDoubleMatrix2D l1l3_OR_R1R3_L1R2,
			SparseDoubleMatrix2D l1l4_OR_R1R4_L2R1) {
		
		int ncols=xtx.columns()+l1l2_OR_R1R2_L1R1.columns()+l1l3_OR_R1R3_L1R2.columns()+l1l4_OR_R1R4_L2R1.columns();
SparseDoubleMatrix2D finalProjection=new SparseDoubleMatrix2D(xtx.rows(),ncols);
		
		for (int i=0;i<xtx.rows();i++){
			for(int j=0; j<xtx.columns();j++){
				finalProjection.set(i, j, xtx.get(i, j));
				finalProjection.set(i, j+xtx.columns(), l1l2_OR_R1R2_L1R1.get(i, j));
				finalProjection.set(i, j+xtx.columns()+l1l2_OR_R1R2_L1R1.columns(), l1l3_OR_R1R3_L1R2.get(i, j));
				finalProjection.set(i, j+xtx.columns()+l1l2_OR_R1R2_L1R1.columns()+l1l3_OR_R1R3_L1R2.columns(), l1l4_OR_R1R4_L2R1.get(i, j));
			}
		}
		return finalProjection;	
	}
	
	public SparseDoubleMatrix2D  concatenateMultiCol(SparseDoubleMatrix2D xtx,
			SparseDoubleMatrix2D l1l2_OR_R1R2_L1R1,
			SparseDoubleMatrix2D l1l3_OR_R1R3_L1R2,
			SparseDoubleMatrix2D l1l4_OR_R1R4_L2R1) {
		
		int nrows=xtx.rows()+l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows()+l1l4_OR_R1R4_L2R1.rows();
SparseDoubleMatrix2D finalProjection=new SparseDoubleMatrix2D(nrows, xtx.columns());
		
		for (int i=0;i<xtx.rows();i++){
			for(int j=0; j<xtx.columns();j++){
				finalProjection.set(i, j, xtx.get(i, j));
				finalProjection.set(i+xtx.rows(),j, l1l2_OR_R1R2_L1R1.get(i, j));
				finalProjection.set(i+xtx.rows()+l1l2_OR_R1R2_L1R1.rows(), j, l1l3_OR_R1R3_L1R2.get(i, j));
				finalProjection.set(i+xtx.rows()+l1l2_OR_R1R2_L1R1.rows()+l1l3_OR_R1R3_L1R2.rows(), j, l1l4_OR_R1R4_L2R1.get(i, j));
			}
		}
		return finalProjection;
		
	}

	///////////
	public FlexCompRowMatrix  concatenateMultiRow(FlexCompRowMatrix xtx,
			FlexCompRowMatrix l1l2_OR_R1R2_L1R1,
			FlexCompRowMatrix l1l3_OR_R1R3_L1R2,
			FlexCompRowMatrix l1l4_OR_R1R4_L2R1) {
		
		int ncols=xtx.numColumns()+l1l2_OR_R1R2_L1R1.numColumns()+l1l3_OR_R1R3_L1R2.numColumns()+l1l4_OR_R1R4_L2R1.numColumns();
		FlexCompRowMatrix finalProjection=new FlexCompRowMatrix(xtx.numRows(),ncols);
		
		for (int i=0;i<xtx.numRows();i++){
			for(int j=0; j<xtx.numColumns();j++){
				finalProjection.set(i, j, xtx.get(i, j));
				finalProjection.set(i, j+xtx.numColumns(), l1l2_OR_R1R2_L1R1.get(i, j));
				finalProjection.set(i, j+xtx.numColumns()+l1l2_OR_R1R2_L1R1.numColumns(), l1l3_OR_R1R3_L1R2.get(i, j));
				finalProjection.set(i, j+xtx.numColumns()+l1l2_OR_R1R2_L1R1.numColumns()+l1l3_OR_R1R3_L1R2.numColumns(), l1l4_OR_R1R4_L2R1.get(i, j));
			}
		}
		return finalProjection;	
	}
	
	public FlexCompRowMatrix  concatenateMultiCol(FlexCompRowMatrix xtx,
			FlexCompRowMatrix l1l2_OR_R1R2_L1R1,
			FlexCompRowMatrix l1l3_OR_R1R3_L1R2,
			FlexCompRowMatrix l1l4_OR_R1R4_L2R1) {
		
		int nrows=xtx.numRows()+l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows()+l1l4_OR_R1R4_L2R1.numRows();
		FlexCompRowMatrix finalProjection=new FlexCompRowMatrix(nrows, xtx.numColumns());
		
		for (int i=0;i<xtx.numRows();i++){
			for(int j=0; j<xtx.numColumns();j++){
				finalProjection.set(i, j, xtx.get(i, j));
				finalProjection.set(i+xtx.numRows(),j, l1l2_OR_R1R2_L1R1.get(i, j));
				finalProjection.set(i+xtx.numRows()+l1l2_OR_R1R2_L1R1.numRows(), j, l1l3_OR_R1R3_L1R2.get(i, j));
				finalProjection.set(i+xtx.numRows()+l1l2_OR_R1R2_L1R1.numRows()+l1l3_OR_R1R3_L1R2.numRows(), j, l1l4_OR_R1R4_L2R1.get(i, j));
			}
		}
		return finalProjection;
		
	}
	
	
	
	public Object[] getDocs() {
		// TODO Auto-generated method stub
		return _allDocs;
	}
	
	/*
	public Matrix getContextOblEmbeddings(Matrix eigenFeatDict) {
		Matrix WProjectionMatrix;
		
		WProjectionMatrix=generateWProjections(_allDocs,_rin.getSortedWordList(),eigenFeatDict);
		
		
		return WProjectionMatrix;
	}
	*/

	
}
