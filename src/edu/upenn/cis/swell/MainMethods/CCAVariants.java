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
import edu.upenn.cis.swell.Runs.CCAVariantsRun;
import edu.upenn.cis.swell.SpectralRepresentations.ContextPCARepresentation;

public class CCAVariants implements Serializable {
 
	static final long serialVersionUID = 42L;
	static HashMap<Integer,Integer> words_Dict;
	
	public static void main(String[] args) throws Exception{
		
		ArrayList<ArrayList<Integer>> all_Docs;
		ArrayList<Integer> docSize;
		ReadDataFile rin;
		ContextPCARepresentation contextPCARep;
		HashMap<String,Integer> corpusInt=new HashMap<String,Integer>();
		HashMap<String,Integer> corpusIntOldMapping=new HashMap<String,Integer>();
		CCAVariantsRun ccaVariantRun;
		ContextPCAWriter wout;
		Object[] matrices=new Object[2];
		long numTokens;
		
		Options opt=new Options(args);
		
		if(opt.algorithm==null){
			System.out.println("WARNING: YOU NEED TO SPECIFY A VALID ALGORITHM NAME AS algorithm:");
		}
		if(opt.trainUnlab){
			System.out.println("+++Inducing CCA Embedddings from unlabeled data+++\n");
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
			
			
			
		    /* Total memory currently in use by the JVM */
		    System.out.println("Total memory (bytes): " + 
		        Runtime.getRuntime().totalMemory());
			
		    
		    
			if(opt.kdimDecomp){
				contextPCARep= new ContextPCARepresentation(opt, numTokens,rin,all_Docs,(Matrix)getkDimCCADict(opt,corpusInt),getwordDict());
			}
			else{
				contextPCARep= new ContextPCARepresentation(opt, numTokens,rin,all_Docs);
			}
		    
		    ccaVariantRun=new CCAVariantsRun(opt,contextPCARep);
			ccaVariantRun.serializeCCAVariantsRun();
			matrices=deserializeCCAVariantsRun(opt);

			wout=new ContextPCAWriter(opt,all_Docs,matrices,rin);
			
			
			wout.writeEigenDict();
			if(!opt.typeofDecomp.equals("TwoStepLRvsW") && !opt.typeofDecomp.equals("LRMVL1") && !opt.kdimDecomp)
				wout.writeEigContextVectors();
			
			
			if (opt.randomBaseline){
				wout.writeEigenDictRandom();
				wout.writeEigContextVectorsRandom();
			}
			
			
		    /* Total memory currently in use by the JVM */
		    System.out.println("Total memory (bytes): " + Runtime.getRuntime().totalMemory());
			
		    if(opt.writeContextMatrix)
				wout.writeSparseMatrix(contextPCARep.getWTLRMatrix(),contextPCARep.getWTWMatrix());
			
			
			System.out.println("+++CCA Embedddings Induced+++\n");
		}
		if (opt.train){
			System.out.println("+++Generating CCA Embedddings for training data+++\n");
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
			matrices=deserializeCCAVariantsRun(opt);
			contextPCARep= new ContextPCARepresentation(opt, numTokens,rin,all_Docs);
			Matrix contextSpecificEmbed;
			contextSpecificEmbed=contextPCARep.generateProjections((Matrix)matrices[0], 
						(Matrix)matrices[1],(Matrix)matrices[2]);
			
			Matrix contextObliviousEmbed=contextPCARep.getContextOblEmbeddings((Matrix)matrices[0]);			
			wout=new ContextPCAWriter(opt,all_Docs,matrices,rin);
			if(opt.typeofDecomp.equals("TwoStepLRvsW") || opt.typeofDecomp.equals("LRMVL1") )
				wout.writeContextSpecificEmbedLRMVL(contextSpecificEmbed);
			else	
				wout.writeContextSpecificEmbed(contextSpecificEmbed);
			wout.writeContextObliviousEmbed(contextObliviousEmbed);
			System.out.println("+++Generated CCA Embedddings for training data+++\n");
		}
	}

	public static HashMap<String,Integer> deserializeCorpusIntMapped(Options opt) throws ClassNotFoundException{
		File f= new File(opt.serializeCorpus);
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
	
public static Matrix getkDimCCADict(Options opt, HashMap<String, Integer> corpusInt) throws ClassNotFoundException{
		
		
		
		//String eigDict=opt.eigenWordCCAFile;
		File fEig= new File(opt.eigenWordCCAFile);
		
		
		Matrix eigDictMat=new Matrix(opt.n+1,opt.p);
		HashMap<Integer,Integer> wordsDict=new HashMap<Integer,Integer>();
		
		try{
			
			BufferedReader reader = new BufferedReader(new FileReader(fEig));
			String line = null;
			int i=0;
			while ((line = reader.readLine()) != null) {
			
			String[] words= line.split("\\s");
			
			if(corpusInt.get(words[0])!=null)
				wordsDict.put(corpusInt.get(words[0]),i);
			
			for(int j=0;j<words.length-1;j++){
				eigDictMat.set(i, j, Double.parseDouble(words[j+1]));
			}
			i++;	
			}
			
			System.out.println("=======Loaded the k-dim CCA Dictionary=======");
			//For words not in dict use OOV
			for(int l=0; l <opt.vocabSize+1;l++)
			{
				if(wordsDict.get(l)==null){
					wordsDict.put(l,0);
				}
			}
			
			setwordDict(wordsDict);
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
		return eigDictMat;
		
	}

public static HashMap<Integer,Integer> getwordDict(){
	return words_Dict;
}

public static HashMap<Integer,Integer> setwordDict(HashMap<Integer,Integer> wDict){
	return words_Dict=wDict;
}
	

	public static Object[] deserializeCCAVariantsRun(Options opt) throws ClassNotFoundException{
		
		Object[] matrixObj =new Object[3];
		
		String contextDict=opt.serializeRun+"Context";
		File fContext= new File(contextDict);
		
		String eigDict=opt.serializeRun+"Eig";
		File fEig= new File(eigDict);
		
		String eigDictL=opt.serializeRun+"EigL";
		File fEigL= new File(eigDictL);
		
		String eigDictR=opt.serializeRun+"EigR";
		File fEigR= new File(eigDictR);
		
		
		Matrix eigDictMat=null,contextDictMat=null,eigDictLMat=null,eigDictRMat=null;
		
		
		try{
			
			ObjectInput cpcaEig=new ObjectInputStream(new FileInputStream(fEig));
			ObjectInput cpcaContext=new ObjectInputStream(new FileInputStream(fContext));
			
			eigDictMat=(Matrix)cpcaEig.readObject();
			contextDictMat=(Matrix)cpcaContext.readObject();	
			
			System.out.println("=======De-serialized the CCA Variant Run=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		matrixObj[0]=(Object)eigDictMat;
		matrixObj[1]=(Object)contextDictMat;
		matrixObj[2]=null;
		
		if(opt.typeofDecomp.equals("TwoStepLRvsW") || opt.typeofDecomp.equals("LRMVL1")){
			
			try{
				
				ObjectInput cpcaEig=new ObjectInputStream(new FileInputStream(fEig));
				ObjectInput cpcaEigL=new ObjectInputStream(new FileInputStream(fEigL));
				ObjectInput cpcaEigR=new ObjectInputStream(new FileInputStream(fEigR));
				
				eigDictMat=(Matrix)cpcaEig.readObject();
				eigDictLMat=(Matrix)cpcaEigL.readObject();
				eigDictRMat=(Matrix)cpcaEigR.readObject();
				
				
				System.out.println("=======De-serialized the CCA Variant Run=======");
			}
			catch (IOException ioe){
				System.out.println(ioe.getMessage());
			}
			matrixObj[0]=(Object)eigDictMat;
			matrixObj[1]=(Object)eigDictLMat;
			matrixObj[2]=(Object)eigDictRMat;
			
			
		}
	
		return matrixObj;
		
	}


	
	
}