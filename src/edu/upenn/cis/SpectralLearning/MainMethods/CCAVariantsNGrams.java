package edu.upenn.cis.SpectralLearning.MainMethods;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

import Jama.Matrix;
import edu.upenn.cis.SpectralLearning.IO.ContextPCANGramsWriter;
import edu.upenn.cis.SpectralLearning.IO.Options;
import edu.upenn.cis.SpectralLearning.IO.ReadDataFile;
import edu.upenn.cis.SpectralLearning.Runs.CCAVariantsNGramsRun;
import edu.upenn.cis.SpectralLearning.SpectralRepresentations.ContextPCANGramsRepresentation;

public class CCAVariantsNGrams implements Serializable {
 
	static final long serialVersionUID = 42L;
	
	
	public static void main(String[] args) throws Exception{
		
		Object[] all_Docs;
		ArrayList<Integer> docSize;
		ReadDataFile rin;
		//Corpus corpus;
		ContextPCANGramsRepresentation contextPCANGramsRep;
		HashMap<String,Integer> corpusInt=new HashMap<String,Integer>();
		HashMap<String,Integer> corpusIntOldMapping=new HashMap<String,Integer>();
		CCAVariantsNGramsRun ccaVariantNGramRun;
		ContextPCANGramsWriter woutNGrams;
		Object[] matrices=new Object[2];
		long numTokens;
		
		Options opt=new Options(args);
		
		if(opt.algorithm==null){
			System.out.println("WARNING: YOU NEED TO SPECIFY A VALID ALGORITHM NAME AS algorithm:");
		}
		if(opt.numGrams ==2 && !opt.typeofDecomp.equals("2viewWvsR")){
			System.out.println("WARNING: WITH BIGRAMS YOU CAN ONLY RUN W vs R; FOR OTHER VARIANTS E.G W vs [L R] TRY 3 OR 5 GRAMS");
			System.exit(0);
		}
		
		
		if(opt.trainUnlab){
			System.out.println("+++Inducing CCA Ngrams Embedddings from unlabeled data+++\n");
			//all_Docs=new HashMap<Double, Integer>();
			docSize=new ArrayList<Integer>();
			rin=new ReadDataFile(opt);
			if (opt.depbigram){
				corpusInt= rin.convertAllDocsIntNGrams();
				all_Docs=rin.readAllDocsNGrams();
			}
			else{
				corpusInt= rin.convertAllDocsIntNGramsSingleVocab();
				all_Docs=rin.readAllDocsNGramsSingleVocab();
			}
			rin.serializeCorpusIntMapped();
			numTokens=rin.getNumTokens();
			rin.serializeCorpusIntMappedContext();
			
			
			contextPCANGramsRep= new ContextPCANGramsRepresentation(opt, numTokens,rin, all_Docs);
			
		    /* Total memory currently in use by the JVM */
		    System.out.println("Total memory (bytes): " + 
		        Runtime.getRuntime().totalMemory());
			
			
			ccaVariantNGramRun=new CCAVariantsNGramsRun(opt,contextPCANGramsRep);
			ccaVariantNGramRun.serializeCCAVariantsNGramsRun();
			matrices=deserializeCCAVariantsNGramsRun(opt);

			woutNGrams=new ContextPCANGramsWriter(opt,all_Docs,matrices,rin);
			woutNGrams.writeEigenDict();
			if(!opt.typeofDecomp.equals("TwoStepLRvsW"))
				woutNGrams.writeEigContextVectors();
			
		    /* Total memory currently in use by the JVM */
		    System.out.println("Total memory (bytes): " + 
		        Runtime.getRuntime().totalMemory());
			
			
			System.out.println("+++CCA NGram Embedddings Induced+++\n");
		}
	
	}

	public static HashMap<String,Integer> deserializeCorpusIntMapped(Options opt) throws ClassNotFoundException{
		File f= new File(opt.serializeCorpus);
		HashMap<String,Integer> corpus_intM=null;
		
		try{
			
			ObjectInput c_intM=new ObjectInputStream(new FileInputStream(f));
			corpus_intM=(HashMap<String,Integer>)c_intM.readObject();
			
			System.out.println("=======De-serialized the CCAVariants NGrams Corpus Int Mapping=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		
		return corpus_intM;
		
	} 

	public static Object[] deserializeCCAVariantsNGramsRun(Options opt) throws ClassNotFoundException{
		
		Object[] matrixObj =new Object[2];
		
		String contextDict=opt.serializeRun+"Context";
		File fContext= new File(contextDict);
		
		String eigDict=opt.serializeRun+"Eig";
		File fEig= new File(eigDict);
		
		
		Matrix eigDictMat=null,contextDictMat=null;
		
		
		try{
			
			ObjectInput ccaEig=new ObjectInputStream(new FileInputStream(fEig));
			ObjectInput ccaContext=new ObjectInputStream(new FileInputStream(fContext));
			
			eigDictMat=(Matrix)ccaEig.readObject();
			contextDictMat=(Matrix)ccaContext.readObject();	
			
			System.out.println("=======De-serialized the CCA Variant NGrams Run=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		matrixObj[0]=(Object)eigDictMat;
		matrixObj[1]=(Object)contextDictMat;
		
		return matrixObj;
		
			
		}

			

	
	
}