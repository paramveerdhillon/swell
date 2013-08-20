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
import edu.upenn.cis.SpectralLearning.IO.ContextPCAWriter;
import edu.upenn.cis.SpectralLearning.IO.Options;
import edu.upenn.cis.SpectralLearning.IO.ReadDataFile;
import edu.upenn.cis.SpectralLearning.Runs.ContextPCANGramsRun;
import edu.upenn.cis.SpectralLearning.SpectralRepresentations.ContextPCANGramsRepresentation;
import edu.upenn.cis.SpectralLearning.SpectralRepresentations.ContextPCARepresentation;

public class ContextPCANGrams implements Serializable {
			static final long serialVersionUID = 42L;
		
		
		public static void main(String[] args) throws Exception{
			
			Object[] all_Docs;
			ReadDataFile rin;
			ContextPCANGramsRepresentation contextPCARepNGrams;
			HashMap<String,Integer> corpusInt=new HashMap<String,Integer>();
			ContextPCANGramsRun contextPCARunNGrams;
			ContextPCANGramsWriter woutNGrams;
			Object[] matrices=new Object[3];
			long numTokens;
			
			Options opt=new Options(args);
			if(opt.algorithm==null){
				System.out.println("WARNING: YOU NEED TO SPECIFY A VALID ALGORITHM NAME AS algorithm:");
			}
			if(opt.numGrams ==2 && !opt.typeofDecomp.equals("WvsR")){
				System.out.println("WARNING: WITH BIGRAMS YOU CAN ONLY RUN W vs R; FOR OTHER VARIANTS E.G W vs [L R] TRY 3 OR 5 GRAMS");
				System.exit(0);
			}
			
			if(opt.trainUnlab){
				System.out.println("+++Inducing Context PCA Embedddings from n-grams data+++\n");
				rin=new ReadDataFile(opt);
				if (opt.depbigram){
					corpusInt= rin.convertAllDocsIntNGrams();
					rin.readAllDocsNGrams();
					all_Docs=rin.getAllDocsNGrams();
				}
				else{
					corpusInt= rin.convertAllDocsIntNGramsSingleVocab();
					rin.readAllDocsNGramsSingleVocab();
					all_Docs=rin.getAllDocsNGrams();
				}
				rin.serializeCorpusIntMapped();
				numTokens=rin.getNumTokens();
				rin.serializeCorpusIntMappedContext();
				contextPCARepNGrams= new ContextPCANGramsRepresentation(opt, numTokens,rin, all_Docs);
				contextPCARunNGrams=new ContextPCANGramsRun(opt,contextPCARepNGrams);
				contextPCARunNGrams.serializeContextPCANGramsRun();
				matrices=deserializeContextPCANGramsRun(opt);

				woutNGrams=new ContextPCANGramsWriter(opt,all_Docs.length,matrices,rin);
				woutNGrams.writeEigenDict();
				woutNGrams.writeEigContextVectors();
				
				if (opt.randomBaseline){
					woutNGrams.writeEigenDictRandom();
					woutNGrams.writeEigContextVectorsRandom();
				}
				
				System.out.println("+++Context PCA NGrams Embedddings Induced+++\n");
			}

		
			
		}

			
		public static HashMap<String,Integer> deserializeCorpusIntMapped(Options opt) throws ClassNotFoundException{
			File f= new File(opt.serializeCorpus);
			HashMap<String,Integer> corpus_intM=null;
			
			try{
				
				ObjectInput c_intM=new ObjectInputStream(new FileInputStream(f));
				corpus_intM=(HashMap<String,Integer>)c_intM.readObject();
				
				System.out.println("=======De-serialized the CPCA NGrams Corpus Int Mapping=======");
			}
			catch (IOException ioe){
				System.out.println(ioe.getMessage());
			}
			
			return corpus_intM;
			
		} 
		
		public static HashMap<String,Integer> deserializeCorpusIntMappedContext(Options opt) throws ClassNotFoundException{
			File f= new File(opt.serializeCorpus+"Context");
			HashMap<String,Integer> corpus_intM=null;
			
			try{
				
				ObjectInput c_intM=new ObjectInputStream(new FileInputStream(f));
				corpus_intM=(HashMap<String,Integer>)c_intM.readObject();
				
				System.out.println("=======De-serialized the CPCA NGrams Corpus Context Int Mapping=======");
			}
			catch (IOException ioe){
				System.out.println(ioe.getMessage());
			}
			
			return corpus_intM;
			
		} 
		

		
public static Object[] deserializeContextPCANGramsRun(Options opt) throws ClassNotFoundException{
		
	Object[] matrixObj =new Object[2];
	
	String contextDict=opt.serializeRun+"Context";
	File fContext= new File(contextDict);
	
	String eigDict=opt.serializeRun+"Eig";
	File fEig= new File(eigDict);
	
	
	Matrix eigDictMat=null,contextDictMat=null;
	
	
	try{
		
		ObjectInput cpcaEig=new ObjectInputStream(new FileInputStream(fEig));
		ObjectInput cpcaContext=new ObjectInputStream(new FileInputStream(fContext));
		
		eigDictMat=(Matrix)cpcaEig.readObject();
		contextDictMat=(Matrix)cpcaContext.readObject();	
		
		System.out.println("=======De-serialized the CPCA NGrams Run=======");
	}
	catch (IOException ioe){
		System.out.println(ioe.getMessage());
	}
	matrixObj[0]=(Object)eigDictMat;
	matrixObj[1]=(Object)contextDictMat;
	
	return matrixObj;
	
		
	}

		
		
	}


