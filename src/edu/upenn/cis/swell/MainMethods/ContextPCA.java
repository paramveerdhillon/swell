package edu.upenn.cis.swell.MainMethods;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */


import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

import Jama.Matrix;
import edu.upenn.cis.swell.IO.ContextPCAWriter;
import edu.upenn.cis.swell.IO.Options;
import edu.upenn.cis.swell.IO.ReadDataFile;
import edu.upenn.cis.swell.Runs.ContextPCARun;
import edu.upenn.cis.swell.SpectralRepresentations.ContextPCARepresentation;

public class ContextPCA implements Serializable {
 
	static final long serialVersionUID = 42L;
	static HashMap<String,Integer> words_Dict;
	static Matrix eigDict=null;
	public static void main(String[] args) throws Exception{
		
		ArrayList<ArrayList<Integer>> all_Docs;
		ArrayList<Integer> docSize;
		ReadDataFile rin;
		ContextPCARepresentation contextPCARep;
		HashMap<String,Integer> corpusInt=new HashMap<String,Integer>();
		HashMap<String,Integer> corpusIntOldMapping=new HashMap<String,Integer>();
		ContextPCARun contextPCARun;
		ContextPCAWriter wout;
		Object[] matrices=new Object[3];
		
		
		long numTokens;
		
		Options opt=new Options(args);
		
		if(opt.algorithm==null){
			System.out.println("WARNING: YOU NEED TO SPECIFY A VALID ALGORITHM NAME AS algorithm:");
		}
		if(opt.trainUnlab){
			System.out.println("+++Inducing Context PCA Embedddings from unlabeled data+++\n");
			all_Docs=new ArrayList<ArrayList<Integer>>();
			docSize=new ArrayList<Integer>();
			rin=new ReadDataFile(opt);
			
			long maxMemory = Runtime.getRuntime().maxMemory();
		    /* Maximum amount of memory the JVM will attempt to use */
		    System.out.println("Maximum memory (bytes): " + 
		        (maxMemory == Long.MAX_VALUE ? "no limit" : maxMemory));

		    /* Total memory currently in use by the JVM */
		    System.out.println("Total memory (bytes): " + 
		        Runtime.getRuntime().totalMemory());
			
			corpusInt= rin.convertAllDocsInt(0);
			rin.readAllDocs(0);
			all_Docs=rin.getAllDocs();
			docSize=rin.getDocSizes();
			numTokens=rin.getNumTokens();
			rin.serializeCorpusIntMapped();
			//corpus=new Corpus(all_Docs,docSize,opt);
			
			contextPCARep= new ContextPCARepresentation(opt, numTokens,rin,all_Docs);
			
		    /* Total memory currently in use by the JVM */
		    System.out.println("Total memory (bytes): " + 
		        Runtime.getRuntime().totalMemory());
			
			
			contextPCARun=new ContextPCARun(opt,contextPCARep);
			contextPCARun.serializeContextPCARun();
			matrices=deserializeContextPCARun(opt);

			wout=new ContextPCAWriter(opt,all_Docs,matrices,rin);
			wout.writeEigenDict();
			wout.writeEigContextVectors();
			
			if (opt.randomBaseline){
				wout.writeEigenDictRandom();
				wout.writeEigContextVectorsRandom();
			}
			
		    /* Total memory currently in use by the JVM */
		    System.out.println("Total memory (bytes): " + Runtime.getRuntime().totalMemory());
			
			System.out.println("+++Context PCA Embedddings Induced+++\n");
		}
		if (opt.train){
			System.out.println("+++Generating Context PCA Embedddings for training data+++\n");
			all_Docs=new ArrayList<ArrayList<Integer>>();
			docSize=new ArrayList<Integer>();
			corpusIntOldMapping=deserializeCorpusIntMapped(opt);
			rin=new ReadDataFile(opt);
			rin.setCorpusIntMapped(corpusIntOldMapping);
			rin.readAllDocs(1);
			all_Docs=rin.getAllDocs();
			docSize=rin.getDocSizes();
			numTokens=rin.getNumTokens();
			//corpus=new Corpus(all_Docs,docSize,opt);
			matrices=deserializeContextPCARun(opt);
			contextPCARep= new ContextPCARepresentation(opt, numTokens,rin,all_Docs);
			wout=new ContextPCAWriter(opt,all_Docs,matrices,rin);
			
			//Matrix contextSpecificEmbed=contextPCARep.generateProjections((Matrix)matrices[0], 
				//	(Matrix)matrices[1], (Matrix)matrices[2]);
			//wout.writeContextSpecificEmbed(contextSpecificEmbed);
			
			Matrix contextObliviousEmbed=contextPCARep.getContextOblEmbeddings((Matrix)matrices[1]);
			wout.writeContextObliviousEmbed(contextObliviousEmbed);
			
			Matrix contextObliviousEmbedContext=contextPCARep.getContextOblEmbeddings((Matrix)matrices[0]);
			wout.writeContextObliviousEmbedContext(contextObliviousEmbedContext);
			
			if (opt.randomBaseline){
				wout.writeContextObliviousEmbedRandom();
			}
			
			System.out.println("+++Generated Context PCA Embeddings for training data+++\n");
		}
		if(opt.induceEmbeds){
			embedMatrixProcess(opt);
			wout=new ContextPCAWriter(opt);
			wout.writeContextObliviousEmbedNewData(getEmbedMatrix(),getwordDict());
		}
		
		
	}
	


	public static HashMap<String,Integer> deserializeCorpusIntMapped(Options opt) throws ClassNotFoundException{
		
		String serCorpus =opt.serializeCorpus;
		
		if (opt.embedToInduce !=null){
			String[] serCorpus1= serCorpus.split("\\.");
			serCorpus= serCorpus1[0]+"."+opt.embedToInduce;
		}
		
		
		File f= new File(serCorpus);
		HashMap<String,Integer> corpus_intM=null;
		
		try{
			
			ObjectInput c_intM=new ObjectInputStream(new FileInputStream(f));
			corpus_intM=(HashMap<String,Integer>)c_intM.readObject();
			
			System.out.println("=======De-serialized the CPCA Corpus Int Mapping=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
		return corpus_intM;
		
	} 

	
public static Matrix embedMatrixProcess(Options opt) throws ClassNotFoundException{
		
		
		
		//String eigDict=opt.eigenWordCCAFile;
		File fEig= new File(opt.eigenEmbedFile);
		
		
		Matrix eigDictMat=new Matrix(opt.n+1,opt.p);
		HashMap<String,Integer> wordsDict=new HashMap<String,Integer>();
		
		try{
			
			BufferedReader reader = new BufferedReader(new FileReader(fEig));
			String line = null;
			int i=0;
			while ((line = reader.readLine()) != null) {
			
			String[] words= line.split("\\s");
			
			wordsDict.put(words[0],i);
			
			for(int j=0;j<words.length-1;j++){
				eigDictMat.set(i, j, Double.parseDouble(words[j+1]));
			}
			i++;	
			}
			
			System.out.println("=======Loaded the k-dim CCA Dictionary=======");
			//For words not in dict use OOV
			/*
			for(int l=0; l <opt.vocabSize+1;l++)
			{
				if(wordsDict.get(l)==null){
					wordsDict.put(l,0);
				}
			}
			*/
			setwordDict(wordsDict);
			setEmbedMatrix(eigDictMat);
		}
		
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
		
		return eigDictMat;
		
	}	
	
public static Matrix setEmbedMatrix(Matrix m){
	return eigDict=m;
}

public static HashMap<String,Integer> setwordDict(HashMap<String,Integer> wDict){
	return words_Dict=wDict;
}

public static Matrix getEmbedMatrix(){
	return eigDict;
}

public static HashMap<String,Integer> getwordDict(){
	return words_Dict;
}

	
	
public static Object[] deserializeContextPCARun(Options opt) throws ClassNotFoundException{
	
	Object[] matrixObj =new Object[2];
	String serRun =opt.serializeRun;
	
	if (opt.embedToInduce !=null){
		String[] serRun1= serRun.split("\\.");
		serRun= serRun1[0]+"."+opt.embedToInduce;
	}
	
	String contextDict=serRun+"Context";
	File fContext= new File(contextDict);
	
	String eigDict=serRun+"Eig";
	File fEig= new File(eigDict);
	
	
	Matrix eigDictMat=null,contextDictMat=null;
	
	
	try{
		
		ObjectInput cpcaEig=new ObjectInputStream(new FileInputStream(fEig));
		ObjectInput cpcaContext=new ObjectInputStream(new FileInputStream(fContext));
		
		eigDictMat=(Matrix)cpcaEig.readObject();
		contextDictMat=(Matrix)cpcaContext.readObject();	
		
		System.out.println("=======De-serialized the CPCA Run=======");
	}
	catch (IOException ioe){
		System.out.println(ioe.getMessage());
	}
	matrixObj[0]=(Object)eigDictMat;
	matrixObj[1]=(Object)contextDictMat;
	
	return matrixObj;
	
}


}
