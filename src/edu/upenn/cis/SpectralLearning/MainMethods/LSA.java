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
import edu.upenn.cis.SpectralLearning.Data.Corpus;
import edu.upenn.cis.SpectralLearning.IO.LSAWriter;
import edu.upenn.cis.SpectralLearning.IO.Options;
import edu.upenn.cis.SpectralLearning.IO.ReadDataFile;
import edu.upenn.cis.SpectralLearning.Runs.LSARun;
import edu.upenn.cis.SpectralLearning.SpectralRepresentations.LSARepresentation;

public class LSA implements Serializable {
	
	static final long serialVersionUID = 42L;
	public static void main(String[] args) throws Exception{
	
		ArrayList<ArrayList<Integer>> all_Docs;
		ArrayList<Integer> docSize;
		ReadDataFile rin;
		//Corpus corpus;
		LSARepresentation lsaRep;
		LSARun lsaRun;
		Object[] matrices=new Object[1];
		HashMap<String,Integer> corpusInt=new HashMap<String,Integer>();
		HashMap<String,Integer> corpusIntOldMapping=new HashMap<String,Integer>();
		LSAWriter wout;
		long numTokens;
		Options opt=new Options(args);
		
		if(opt.algorithm==null){
			System.out.println("WARNING: YOU NEED TO SPECIFY A VALID ALGORITHM NAME AS algorithm:");
		}
		
		if(opt.trainUnlab){
		
			System.out.println("+++Inducing LSA Embedddings from unlabeled data+++\n");
			all_Docs=new ArrayList<ArrayList<Integer>>();
			docSize=new ArrayList<Integer>();
			rin=new ReadDataFile(opt);
			corpusInt= rin.convertAllDocsInt(0);
			rin.readAllDocs(0);
			all_Docs=rin.getAllDocs();
			docSize=rin.getDocSizes();
			numTokens=rin.getNumTokens();
			rin.serializeCorpusIntMapped();
			//corpus=new Corpus(all_Docs,docSize,opt);
			//corpus.CreateIntMapping();
			//corpus.updateDocsWithInts(corpus.getCorpusAllWords(), corpus.getCorpusIntMapped());
			//corpus.serializeCorpus();

			lsaRep= new LSARepresentation(opt, numTokens,rin,all_Docs);
			//lsaRep.serializeLSARepresentation();
			lsaRun=new LSARun(opt,lsaRep,all_Docs);
			lsaRun.serializeLSARun();
			matrices=deserializeLSARun(opt);
			wout=new LSAWriter(opt,all_Docs,(Matrix)matrices[0],rin);
			wout.writeEigenDict();
			//wout.writeDocDictL();
			System.out.println("+++LSA Embedddings Induced+++\n");
			
			if (opt.randomBaseline){
				wout.writeEigenDictRandom();
			}
			
		}
		if (opt.train){
			System.out.println("+++Generating LSA Embedddings for training data+++\n");
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
			matrices=deserializeLSARun(opt);
			lsaRep= new LSARepresentation(opt, numTokens,rin,all_Docs);
			
			
			Matrix contextObliviousEmbed=lsaRep.getContextOblEmbeddings((Matrix)matrices[0]);

			wout=new LSAWriter(opt,all_Docs,(Matrix)matrices[0],rin);
			
			
			wout.writeContextObliviousEmbed(contextObliviousEmbed);
			
			if (opt.randomBaseline){
				wout.writeContextObliviousEmbedRandom();
			}
			
			System.out.println("+++Generated LSA Embedddings for training data+++\n");
		}
	
	}
		
		public static HashMap<String,Integer> deserializeCorpusIntMapped(Options opt) throws ClassNotFoundException{
			File f= new File(opt.serializeCorpus);
			HashMap<String,Integer> corpus_intM=null;
			
			try{
				
				ObjectInput c_intM=new ObjectInputStream(new FileInputStream(f));
				corpus_intM=(HashMap<String,Integer>)c_intM.readObject();
				
				System.out.println("=======De-serialized the LSA Corpus Int Mapping=======");
			}
			catch (IOException ioe){
				System.out.println(ioe.getMessage());
			}
			
			return corpus_intM;
			
		} 

	public static Object[] deserializeLSARun(Options opt) throws ClassNotFoundException{
		
		Object[] matrixObj=new Object[1];
		String eigDict=opt.serializeRun;
		File fEig= new File(eigDict);
		
		Matrix eig_Dict=null;
		
		
		try{
			
			ObjectInput ccaw=new ObjectInputStream(new FileInputStream(fEig));
			
			eig_Dict=(Matrix)ccaw.readObject();
			
			System.out.println("=======De-serialized the LSA Run=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
		matrixObj[0]=(Object)eig_Dict;
		
		return matrixObj;
		
	}

	
	
}
