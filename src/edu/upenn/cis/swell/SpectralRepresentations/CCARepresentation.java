package edu.upenn.cis.swell.SpectralRepresentations;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */


import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import Jama.Matrix;
import edu.upenn.cis.swell.IO.Options;
import edu.upenn.cis.swell.IO.ReadDataFile;
import edu.upenn.cis.swell.MathUtils.CenterScaleNormalizeUtils;

public class CCARepresentation extends SpectralRepresentation implements Serializable {

	protected ArrayList<Double> _smooths=new ArrayList<Double>();
	protected Matrix covLLAllDocsMatrix,covRRAllDocsMatrix,covLRAllDocsMatrix,covRLAllDocsMatrix;
	private ReadDataFile _rin;
	Matrix newEigenDict=null;
	long _numTokens=0;
	static final long serialVersionUID = 42L;
	ArrayList<ArrayList<Integer>> _allDocs;
	
	public CCARepresentation(Options opt, long numTok, ReadDataFile rin, ArrayList<ArrayList<Integer>> all_Docs) {
		super(opt,numTok);
		_smooths=opt.smoothArray;
		_rin=rin;
		_numTokens=numTok;
		_allDocs=all_Docs;
		double[][] covLLAllDocs=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
		double[][] covRRAllDocs=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
		double[][] covLRAllDocs=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
		double[][] covRLAllDocs=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
		
		covLLAllDocsMatrix= new Matrix(covLLAllDocs); 
		covRRAllDocsMatrix= new Matrix(covRRAllDocs);
		covLRAllDocsMatrix= new Matrix(covLRAllDocs);
		covRLAllDocsMatrix= new Matrix(covRLAllDocs);
		
	}

	public void generateCovForAllDocs(){
		
		
		try {
			processInputs(super.eigenFeatDictMatrix);
			
			
				
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ExecutionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
			
		
		setCovLLAllDocsMatrix(addRegularization(getCovLLAllDocsMatrix()));
		setCovRRAllDocsMatrix(addRegularization(getCovRRAllDocsMatrix()));
		
			
	}	
	
	public Matrix generateProjectionsBySmoothing(Matrix eigenFeatDict, Matrix eigenFeatDictL,Matrix eigenFeatDictR){
		Matrix LTrainSmoothedDocMatrix,RTrainSmoothedDocMatrix,LTrainSmoothedAllDocsMatrix,RTrainSmoothedAllDocsMatrix;
		Matrix LProjectionMatrix,RProjectionMatrix,WProjectionMatrix;
		double[][] LTrainSmoothedAllDocs=new double[(int) _numTokens][_smooths.size()*_num_hidden];
		double[][] RTrainSmoothedAllDocs=new double[(int) _numTokens][_smooths.size()*_num_hidden];
		
		Matrix finalProjectionMatrix=null;
		
		ArrayList<Integer> doc;
		int idx=0,count=0,idxDocs=0;
		
			while(idxDocs<_allDocs.size()){
				
				doc=_allDocs.get(idxDocs++);
				idx=count;
				LTrainSmoothedDocMatrix=left_smooth(eigenFeatDict,doc);
				RTrainSmoothedDocMatrix=right_smooth(eigenFeatDict,doc);
				for(int i=0;i<doc.size();i++){
					count++;
					for(int j=0;j<_smooths.size()*_num_hidden;j++){
						LTrainSmoothedAllDocs[i+idx][j]=LTrainSmoothedDocMatrix.get(i,j);
						RTrainSmoothedAllDocs[i+idx][j]=RTrainSmoothedDocMatrix.get(i,j);
					}
				}
			}
		
		LTrainSmoothedAllDocsMatrix= new Matrix(LTrainSmoothedAllDocs);
		RTrainSmoothedAllDocsMatrix= new Matrix(RTrainSmoothedAllDocs);
		
		LProjectionMatrix=LTrainSmoothedAllDocsMatrix.times(eigenFeatDictL);
		RProjectionMatrix=RTrainSmoothedAllDocsMatrix.times(eigenFeatDictR);
		
		WProjectionMatrix=generateWProjections(_rin.getSortedWordList(),eigenFeatDict);
		
		finalProjectionMatrix=concatenateProjections(LProjectionMatrix,WProjectionMatrix,RProjectionMatrix);
		
		return finalProjectionMatrix;
		
	}
	
	

	private Matrix concatenateProjections(Matrix lProjectionMatrix,
			Matrix wProjectionMatrix, Matrix rProjectionMatrix) {
		double[][] finalProjection=new double[(int)_numTokens][_num_hidden*3];
		
		for (int i=0;i<(int)_numTokens;i++){
			for(int j=0; j<_num_hidden;j++){
				finalProjection[i][j]=lProjectionMatrix.get(i, j);
				finalProjection[i][j+_num_hidden]=wProjectionMatrix.get(i, j);
				finalProjection[i][j+ (2*_num_hidden)]=rProjectionMatrix.get(i, j);
			}
		}
		
		
		return new Matrix(finalProjection);
	}

	public Object[] generateCovForOneDoc(Matrix eigenFeatDict, ArrayList<Integer> doc){
		
		
		double[][] covLL=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
		double[][] covRR=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
		double[][] covLR=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
		double[][] covRL=new double[_smooths.size()*_num_hidden][_smooths.size()*_num_hidden];
		Object[] covMatrices= new Object[4];
		
		mathUtils=new CenterScaleNormalizeUtils(_opt);
		
		
		Matrix covLLMatrix= new Matrix(covLL); 
		Matrix covRRMatrix= new Matrix(covRR);
		Matrix covLRMatrix= new Matrix(covLR);
		Matrix covRLMatrix= new Matrix(covRL);
		
		Matrix LMatrix=mathUtils.center(left_smooth(eigenFeatDict, doc));
		Matrix RMatrix=mathUtils.center(right_smooth(eigenFeatDict, doc));
		
		//Matrix LMatrix=mathUtils.center_and_scale(left_smooth(eigenFeatDict, doc));
		//Matrix RMatrix=mathUtils.center_and_scale(right_smooth(eigenFeatDict, doc));
		
		
		//Matrix LMatrix=left_smooth(eigenFeatDict, doc);
		//Matrix RMatrix=right_smooth(eigenFeatDict, doc);
		
		
		covLLMatrix=addNormalize(LMatrix.transpose().times(LMatrix),LMatrix.getRowDimension());
		covRRMatrix=addNormalize(RMatrix.transpose().times(RMatrix),RMatrix.getRowDimension());
		covLRMatrix=addNormalize(LMatrix.transpose().times(RMatrix),LMatrix.getRowDimension());
		covRLMatrix=addNormalize(RMatrix.transpose().times(LMatrix),RMatrix.getRowDimension());
		
		covMatrices[0]=(Object)covLLMatrix;
		covMatrices[1]=(Object)covRRMatrix;
		covMatrices[2]=(Object)covLRMatrix;
		covMatrices[3]=(Object)covRLMatrix;
		
		return covMatrices;
		
	}
	
	
	
	public Matrix left_smooth(Matrix eigenFeatDict, ArrayList<Integer> doc){
		double[][] L=new double[doc.size()][_smooths.size()*_num_hidden];
		double[][] smoothedStateofToken=new double[1][_smooths.size()*_num_hidden];
		double[][] smoothedStateofTokenCopy=new double[1][_smooths.size()*_num_hidden];
		
		//System.out.println("Doc Size, #sent, #tok="+doc.size()+" "+doc.getNumSentences()+" "+doc.getNumTokens());
		Matrix LMatrix= new Matrix(L); 
		Matrix smoothedStateofTokenMatrix; 
		int tok_idx=1;
		
		while(tok_idx<doc.size()){
			//Token tok=doc.get(tok_idx++);
			//assert counter== tok_idx;
			if (tok_idx==1){
				smoothedStateofTokenCopy=getStateforToken(eigenFeatDict,doc.get(tok_idx-1));//s1=x0
				for (int idx=0; idx<_smooths.size();idx++){
					for (int k=0;k<_num_hidden;k++)
						smoothedStateofToken[0][idx*_num_hidden+k]= smoothedStateofTokenCopy[0][k];
				}
				smoothedStateofTokenMatrix= new Matrix(smoothedStateofToken); 
			}
			else{
				smoothedStateofToken=getNextStateSmoothed(smoothedStateofToken,getStateforToken(eigenFeatDict,doc.get(tok_idx-1)));
				smoothedStateofTokenMatrix= new Matrix(smoothedStateofToken); 
			}
			
			
			LMatrix.setMatrix(tok_idx, tok_idx, 0, LMatrix.getColumnDimension()-1, smoothedStateofTokenMatrix);
			tok_idx++;
			
			
			
			
		}
		
		return LMatrix;
	}
	
	public void left_right_smooths_W(Matrix eigenFeatDict, Matrix left_singular_vectors,Matrix right_singular_vectors){
		
		ArrayList<Integer> countsList; 
		double[][] nEigenDict=new double[_opt.vocabSize+1][_num_hidden];
		
		
		int num_OOV=(int)_numTokens-_opt.vocabSize;
		if (num_OOV <=0)
			num_OOV=1;//Prevent Divide by zero errors.
		
		
		try {
			createNewEigDict();
			processDocsParallel(eigenFeatDict,left_singular_vectors,right_singular_vectors);
			//processDocsSerial(eigenFeatDict,left_singular_vectors,right_singular_vectors);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ExecutionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		countsList=_rin.getSortedCountList();
		nEigenDict=newEigenDict.getArray();
		
		for (int i=1;i<=_opt.vocabSize;i++){ //leave OOV
			for (int j=0;j<_num_hidden;j++){
				nEigenDict[i][j]/=countsList.get(i-1);
			}
		}
		
		
		for (int j=0;j<_num_hidden;j++)
			nEigenDict[0][j]/=num_OOV;
		
	
		setEigenFeatDict(new Matrix(nEigenDict));
	}
	
	
	
	private void processDocsParallel(final Matrix eigenFeatDict, final Matrix left_singular_vectors,final Matrix right_singular_vectors)
	throws InterruptedException, ExecutionException{
		
			int threads = Runtime.getRuntime().availableProcessors();
			//System.out.println(threads);
			//threads=4;
			ExecutorService service = Executors.newFixedThreadPool(threads);
			List<Future<Integer>> futures = new ArrayList<Future<Integer>>();
			final Iterator<ArrayList<Integer>> it= _allDocs.iterator();
			int k=0;
			while (it.hasNext()) {
				Callable<Integer> callable = new Callable<Integer>() {
					
					ArrayList<Integer> _doc=it.next();
					public Integer call() throws Exception {
						Matrix eigDict=left_right_smooths_doc(_doc,eigenFeatDict, left_singular_vectors,right_singular_vectors);
						update_dict(eigDict);
					
						return 1;
					}

				};
				//System.out.println(k++);
				futures.add(service.submit(callable));
				
			}

			service.shutdown();
			List<Integer> outputs = new ArrayList<Integer>();
			int i=0;
		    for (Future<Integer> future : futures) {
		        outputs.add(future.get());
		        i++;
		        if(i%10==0)
		        	System.out.println("+++Doc Num "+(i)+" Processed for Smoothing++");
		    }
		    System.out.println("++Doc Processing for Smoothing Finished++");   
		}
	
/*	
	private void processDocsSerial(final Matrix eigenFeatDict, final Matrix left_singular_vectors,final Matrix right_singular_vectors)
			throws InterruptedException, ExecutionException{
				
					final Iterator<ArrayList<Integer>> it= _allDocs.iterator();
					int i=0;
					while (it.hasNext()) {
								Matrix eigDict=left_right_smooths_doc(it.next(),eigenFeatDict, left_singular_vectors,right_singular_vectors);
								update_dict(eigDict);
						        i++;
						        if(i%1000==0)
						        	System.out.println("+++Doc Num "+(i)+" Processed for Smoothing++");
								}
						//System.out.println(k++);
									
				    				}
*/		
		
	public Matrix createNewEigDict(){
		newEigenDict=new Matrix(new double[_opt.vocabSize+1][_num_hidden]);
		return newEigenDict;
	}
	
	
	private synchronized Matrix update_dict(Matrix eigDict1){
		return newEigenDict.plusEquals(eigDict1);
	}
		
				
		

	private Matrix left_right_smooths_doc(ArrayList<Integer> docs,Matrix eigenFeatDict, Matrix left_singular_vectors,Matrix right_singular_vectors) {
		
		int counter=0,tok_idx=1;
		double[][] newEigenDict=new double[_opt.vocabSize+1][_num_hidden];
		double[][] temp=new double[1][_num_hidden/2];
		double[][] smoothedStateofToken=new double[1][_smooths.size()*_num_hidden];
		double[][] smoothedStateofTokenCopy=new double[1][_smooths.size()*_num_hidden];
		Matrix tempMatrix= new Matrix(temp); 
	
		Matrix smoothedStateofTokenMatrix; 
		
		
		while(tok_idx<docs.size()){
			
			if (tok_idx==1){
				smoothedStateofTokenCopy=getStateforToken(eigenFeatDict,docs.get(tok_idx-1));
				for (int idx=0; idx<_smooths.size();idx++){
					for (int k=0;k<_num_hidden;k++)
						smoothedStateofToken[0][idx*_num_hidden+k]= smoothedStateofTokenCopy[0][k];
				}
				smoothedStateofTokenMatrix= new Matrix(smoothedStateofToken); 
			}
			else{
				smoothedStateofToken=getNextStateSmoothed(smoothedStateofToken,getStateforToken(eigenFeatDict,docs.get(tok_idx-1)));
				smoothedStateofTokenMatrix= new Matrix(smoothedStateofToken); 
			}

			tempMatrix=smoothedStateofTokenMatrix.times(left_singular_vectors);

			for(int j=0;j< tempMatrix.getColumnDimension();j++)
				newEigenDict[docs.get(tok_idx-1)][j]+=tempMatrix.get(0, j);

			//counter++;
			tok_idx++;
		}
		///////////////////////
		counter=docs.size()-2;
		
		while(counter>=0){


			if (counter==docs.size()-2){
				smoothedStateofTokenCopy=getStateforToken(eigenFeatDict,docs.get(counter+1));
				for (int idx=0; idx<_smooths.size();idx++){
					for (int k=0;k<_num_hidden;k++)
						smoothedStateofToken[0][idx*_num_hidden+k]= smoothedStateofTokenCopy[0][k];
				}
				smoothedStateofTokenMatrix= new Matrix(smoothedStateofToken); 
			}
			else{
				smoothedStateofToken=getNextStateSmoothed(smoothedStateofToken,getStateforToken(eigenFeatDict,docs.get(counter+1)));
				smoothedStateofTokenMatrix= new Matrix(smoothedStateofToken); 
			}

			tempMatrix=smoothedStateofTokenMatrix.times(right_singular_vectors);

			for(int j=0;j< tempMatrix.getColumnDimension();j++)
				newEigenDict[docs.get(counter+1)][j+(_num_hidden/2)]+=tempMatrix.get(0, j);
			counter--;

	}

		Matrix m = new Matrix(newEigenDict);
		//addNormalize(m,_num_hidden)
		
		return m;
		
	}

	public Matrix addRegularization(Matrix cov){
		
		double epsilon=0.05;
		double[][] covArr=cov.getArray();
		
		for (int i=0;i<cov.getRowDimension();i++)
			covArr[i][i]+=epsilon;
		/*
		for (int i=0;i<cov.getRowDimension();i++){
			for (int j=0;j<cov.getColumnDimension();j++)
				covArr[i][j]/=_numTokens;
		}
		*/
			
		Matrix covNew= new Matrix(covArr);
		 
		return covNew;
	}
	
public Matrix addNormalize(Matrix cov,int numRows){
		
		double[][] covArr=cov.getArray();
		
		for (int i=0;i<cov.getRowDimension();i++){
			for (int j=0;j<cov.getColumnDimension();j++)
				covArr[i][j]/=numRows;
		}
			
		Matrix covNew= new Matrix(covArr);
		 
		return covNew;
	}
	
	
	public Matrix right_smooth(Matrix eigenFeatDict, ArrayList<Integer> doc){
		double[][] R=new double[doc.size()][_smooths.size()*_num_hidden];
		double[][] smoothedStateofToken=new double[1][_smooths.size()*_num_hidden];
		double[][] smoothedStateofTokenCopy=new double[1][_smooths.size()*_num_hidden];
		Matrix RMatrix= new Matrix(R); 
		Matrix smoothedStateofTokenMatrix; 
		int counter=doc.size()-2;//See exponential smoothing definition. s0=0, s1=x0 etc.
		//Iterator<Token> tok_iter= doc.ReverseIterator();
		
		while(counter>=0){
			
			//Token tok=doc.get(counter);
			//assert counter== tok.getidx();
		
			if (counter==doc.size()-2){
				smoothedStateofTokenCopy=getStateforToken(eigenFeatDict,doc.get(counter+1));
				for (int idx=0; idx<_smooths.size();idx++){
					for (int k=0;k<_num_hidden;k++)
						smoothedStateofToken[0][idx*_num_hidden+k]= smoothedStateofTokenCopy[0][k];
				}
				smoothedStateofTokenMatrix= new Matrix(smoothedStateofToken); //s1=x0
			}
			else{
				smoothedStateofToken=getNextStateSmoothed(smoothedStateofToken,getStateforToken(eigenFeatDict,doc.get(counter+1)));
				smoothedStateofTokenMatrix= new Matrix(smoothedStateofToken); 
			}
			
			RMatrix.setMatrix(counter, counter, 0, RMatrix.getColumnDimension()-1, smoothedStateofTokenMatrix);
			counter--;
			
			
		}
		
		return RMatrix;
	}
	
	public double[][] getStateforToken(Matrix eigenFeatDict,int i){
		return eigenFeatDict.getMatrix(i, i,0,eigenFeatDict.getColumnDimension()-1).getArray();  
	}
	
	public double[][] getNextStateSmoothed(double[][] smoothedState, double[][] eigenDictForToken){
		double[][] smoothedStateOutput=new double[1][_smooths.size()*_num_hidden];
		
		for (int loopIdx=0;loopIdx<_smooths.size();loopIdx++)
		{	
			for (int idx=0;idx<_num_hidden;idx++){
				int modIdxLow=(loopIdx)*_num_hidden;
				smoothedStateOutput[0][modIdxLow+idx]= (1-_smooths.get(loopIdx))*smoothedState[0][modIdxLow+idx]  + _smooths.get(loopIdx) *eigenDictForToken[0][idx];
			}   
		}
		
		return smoothedStateOutput;
		
	}

	public void setCovLLAllDocsMatrix(Matrix covLLAllDocsMatrix) {
		this.covLLAllDocsMatrix = covLLAllDocsMatrix;
	}

	public Matrix getCovLLAllDocsMatrix() {
		return covLLAllDocsMatrix;
	}

	public void setCovRRAllDocsMatrix(Matrix covRRAllDocsMatrix) {
		this.covRRAllDocsMatrix = covRRAllDocsMatrix;
	}

	public Matrix getCovRRAllDocsMatrix() {
		return covRRAllDocsMatrix;
	}

	public void setCovLRAllDocsMatrix(Matrix covLRAllDocsMatrix) {
		this.covLRAllDocsMatrix = covLRAllDocsMatrix;
	}

	public Matrix getCovLRAllDocsMatrix() {
		return covLRAllDocsMatrix;
	}
	
	public void setCovRLAllDocsMatrix(Matrix covRLAllDocsMatrix) {
		this.covRLAllDocsMatrix = covRLAllDocsMatrix;
	}

	public Matrix getCovRLAllDocsMatrix() {
		return covRLAllDocsMatrix;
	}
	
	
	public void processInputs(final Matrix eigenFeatDict)
	throws InterruptedException, ExecutionException {

		int threads = Runtime.getRuntime().availableProcessors();
		ExecutorService service = Executors.newFixedThreadPool(threads);
		List<Future<Integer>> futures = new ArrayList<Future<Integer>>();
		final Iterator<ArrayList<Integer>> it =_allDocs.iterator();
		
		while (it.hasNext()) {
			Callable<Integer> callable = new Callable<Integer>() {
				ArrayList<Integer> aDoc=it.next();
				
				
				
				public Integer call() throws Exception {
					Object[] covMs=null;
					covMs=generateCovForOneDoc(eigenFeatDict, aDoc);
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
	        if(i%10==0)
	        	System.out.println("===Doc Num "+(i)+" Processed===");
	    }
	    System.out.println("===Doc Processing Finished===");
	}

	private synchronized void updateCovs(Object[] covMs) {
		setCovLLAllDocsMatrix(getCovLLAllDocsMatrix().plusEquals((Matrix)covMs[0]));
		setCovRRAllDocsMatrix(getCovRRAllDocsMatrix().plusEquals((Matrix)covMs[1]));
		setCovLRAllDocsMatrix(getCovLRAllDocsMatrix().plusEquals((Matrix)covMs[2]));
		setCovRLAllDocsMatrix(getCovRLAllDocsMatrix().plusEquals((Matrix)covMs[3]));
		
	}
	
	
	public void serializeCCARepresentation(){
		File f= new File(_opt.serializeRep);
		
		try{
			ObjectOutput ccaRep=new ObjectOutputStream(new FileOutputStream(f));
			ccaRep.writeObject(this);
			
			System.out.println("=======Serialized the CCA Representation=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
	}
	
	
	public CCARepresentation deserializeCCARepresentation() throws ClassNotFoundException{
		File f= new File(_opt.serializeRep);
		CCARepresentation ccaR_deserialize=null;
		
		try{
			
			ObjectInput ccaRep=new ObjectInputStream(new FileInputStream(f));
			ccaR_deserialize=(CCARepresentation)ccaRep.readObject();
			
			System.out.println("=======De-serialized the CCA Representation=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
		return ccaR_deserialize;
		
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

	public ArrayList<ArrayList<Integer>> getAllDocs() {
		
		return _allDocs;
	}
	
}

