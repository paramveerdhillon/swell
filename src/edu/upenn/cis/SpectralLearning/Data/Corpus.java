package edu.upenn.cis.SpectralLearning.Data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;


import edu.upenn.cis.SpectralLearning.IO.Options;

public class Corpus implements Serializable {

	
	private HashMap<Integer,Document> corpusAllDocs= new HashMap<Integer,Document>();
	static final long serialVersionUID = 42L;
	private  int _numTokensCorpus=0;
	private int _num_docs=0;
	
	
	public Corpus(ArrayList<ArrayList<Integer>> corpusInt,ArrayList<Integer> docSizes,Options opt){
		Iterator <ArrayList<Integer>> itAllWordsDocs= corpusInt.iterator();
				
		int i=0;
		System.out.println("Size of Doc List "+corpusInt.size()+"\n");
		while (itAllWordsDocs.hasNext()){
		
				Document doc= new Document(itAllWordsDocs.next());
				corpusAllDocs.put(i, doc);
				doc.setNumSentences(docSizes.get(i++));
				System.out.println("+++Read document number+++ "+i+"\n");
				_num_docs++;
				 
				_numTokensCorpus+=doc.getNumTokens();
			}
		
		System.out.println("+++Read a total of "+_numTokensCorpus+" tokens from "+_num_docs+" documents+++");
	}
	
	public Corpus(HashMap<Double,Integer> corpusInt,Options opt){
	
		
		int i=0;
		System.out.println("Size of Doc List "+corpusInt.size()+"\n");
		Document doc= new Document(corpusInt);
		corpusAllDocs.put(i, doc);//Only 1 doc for the n-gram case.

		
	}
	
	
	
	
	public int getNumTokensCorpus(){
		return _numTokensCorpus;
	}
	

	public Iterator<Document> iterator(){
		return corpusAllDocs.values().iterator();
	}
	
	public Collection<Document> loopDocs(){
		return corpusAllDocs.values();
	}
	
	
	public HashMap<Integer,Document> getCorpusAllWords(){
		return corpusAllDocs;
	}
	
		
	
	public int getNumDocs()
	{
		return _num_docs;
	}
	
	public int getSize()
	{
		return corpusAllDocs.size();
	}
	
	public Document getDoc(int i)
	{
		return corpusAllDocs.get(i);
	}
		
	

}
