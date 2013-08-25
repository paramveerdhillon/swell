package edu.upenn.cis.swell.Data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;


public class Document implements Serializable{
	
	
	private HashMap<Integer,Integer> docHMap= new HashMap<Integer,Integer>();
	
	private HashMap<Double,Integer> docHMap_Ngrams= new HashMap<Double,Integer>();
	static final long serialVersionUID = 42L;
	
	private int _numTokens=-1;
	private int _numSentences=-1;
	private int _num_id=-1;
	
	public Document(ArrayList<Integer> docWords,int numSentences){
		 this(docWords);
		_numSentences=numSentences;		
		}
	
	public Document(HashMap<Double,Integer> docWords){
		this.docHMap_Ngrams=docWords;
		}
	
	public Document(ArrayList<Integer> docWords) {
		Iterator<Integer> itDoc= docWords.iterator();
		int i=0;
		while (itDoc.hasNext()){ //Add L contexts
			int wInt=itDoc.next();
			//Token tok= new Token();
			docHMap.put(i++, wInt);	
	}
		
		_numTokens=docWords.size();
	}

	

	public HashMap<Double, Integer> getHashMapNGram(){
		return this.docHMap_Ngrams;
	}
	
	public int getNumSentences(){
		return _numSentences;
	}
	
	public int getNumTokens(){
		return _numTokens;
	}
	
	public void setNumSentences(int numS){
		_numSentences=numS;
	}
	
	
	public int getId(){
		return _num_id;
	}
	
	public void setId(int _id){
		_num_id=_id;
	}
	
	
	public void setNumTokens(int numT){
		 _numTokens=numT;
	}

	public int size() {
        return docHMap.size(); 
	}
	
	public int get(int j) {
        return docHMap.get(j); 
	}
	

	

}
