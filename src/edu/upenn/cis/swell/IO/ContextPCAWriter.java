package edu.upenn.cis.swell.IO;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.StringTokenizer;

import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.tfloat.impl.DenseFloatMatrix2D;
import cern.colt.matrix.tint.impl.DenseIntMatrix2D;
import Jama.Matrix;

public class ContextPCAWriter extends WriteDataFile implements EmbeddingWriter {

	private Object[] _matrices=new Object[3];
	private BufferedWriter writer;
	private ReadDataFile _rin;
	ArrayList<ArrayList<Integer>> _allDocs;
	
	public ContextPCAWriter(Options opt, ArrayList<ArrayList<Integer>> all_Docs,Object[] matrices,ReadDataFile rin) {
		super(opt, all_Docs);
		_allDocs =all_Docs;
		_matrices=matrices;
		_rin=rin;
		
	}
	public ContextPCAWriter(Options opt) {
		super(opt);
}
	
	
	private DenseDoubleMatrix2D createDenseMatrixCOLT(Matrix xJama) {
		DenseDoubleMatrix2D x_omega=new DenseDoubleMatrix2D(xJama.getRowDimension(),xJama.getColumnDimension());
		for (int i=0;i<xJama.getRowDimension();i++){
			for (int j=0;j<xJama.getColumnDimension();j++){
				x_omega.set(i, j, xJama.get(i, j));
			}
		}
		return x_omega;
	}
	
	
	public void writeContextObliviousEmbed(Matrix contextObliviousEmbed) throws IOException {
		int i=0,idxDoc=0,idx=0;
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.contextOblEmbed),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	
		
	
	while(idxDoc<_allDocs.size()){	
			int tok_idx=0;	
			ArrayList<Integer> doc=_allDocs.get(idxDoc++);
			
			double maxInThisDimension= 0;
			while(tok_idx<doc.size()){
				double arr=0;
				maxInThisDimension= 0;
				writer.write(_rin.getTokForIntTrain(idx++));
				writer.write(' ');
				
				for(int k=0;k<_opt.hiddenStateSize;k++){
					arr=contextObliviousEmbed.get(i, k);
					if(maxInThisDimension<Math.abs(arr))
						maxInThisDimension=Math.abs(arr);
				}
				
				for (int j=0;j<_opt.hiddenStateSize;j++){
					
					if ( j != (_opt.hiddenStateSize)-1){
						writer.write(Double.toString(contextObliviousEmbed.get(i, j)/maxInThisDimension));
						writer.write(' ');
					}
					else{
						writer.write(Double.toString(contextObliviousEmbed.get(i, j)/maxInThisDimension));
						writer.write('\n');
					}
				}
				i++;
				tok_idx++;
			}
	}
		writer.close();	
	}
	
	
	public void writeContextObliviousEmbedContext(Matrix contextObliviousEmbed) throws IOException {
		int i=0,idxDoc=0,idx=0;
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.contextOblEmbedContext),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
	
	while(idxDoc<_allDocs.size()){	
			int tok_idx=0;	
			
			ArrayList<Integer> doc=_allDocs.get(idxDoc++);
			
			double maxInThisDimension= 0;
			
			while(tok_idx<doc.size()){
				double arr=0;
				maxInThisDimension= 0;
				writer.write(_rin.getTokForIntTrain(idx++));
				writer.write(' ');
				
				for(int k=0;k<_opt.hiddenStateSize;k++){
					arr=contextObliviousEmbed.get(i, k);
					if(maxInThisDimension<Math.abs(arr))
						maxInThisDimension=Math.abs(arr);
				}
				
				for (int j=0;j<_opt.hiddenStateSize;j++){
					
					if ( j != (_opt.hiddenStateSize)-1){
						writer.write(Double.toString(contextObliviousEmbed.get(i, j)/maxInThisDimension));
						writer.write(' ');
					}
					else{
						writer.write(Double.toString(contextObliviousEmbed.get(i, j)/maxInThisDimension));
						writer.write('\n');
					}
				}
				i++;
				tok_idx++;
			}
	}
		writer.close();	
	}
	
	
	public void writeContextObliviousEmbedRandom() throws IOException {
		int i=0,idxDoc=0,idx=0;
		Random r =new Random();
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.contextOblEmbed+"Random"),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
	
	while(idxDoc<_allDocs.size()){	
			int tok_idx=0;	
			ArrayList<Integer> doc=_allDocs.get(idxDoc++);
			
			while(tok_idx<doc.size()){
				writer.write(_rin.getTokForIntTrain(idx++));
				writer.write(' ');
				for (int j=0;j<_opt.hiddenStateSize;j++){
					
					if ( j != (_opt.hiddenStateSize)-1){
						writer.write(Double.toString(r.nextGaussian()));
						writer.write(' ');
					}
					else{
						writer.write(Double.toString(r.nextGaussian()));
						writer.write('\n');
					}
				}
				i++;
				tok_idx++;
			}
	}
		writer.close();	
	}
	
	public void writeEigContextVectors() throws IOException{
		Matrix eigenDictContext=(Matrix)_matrices[1];
		double[][] eigenDictArrContext=eigenDictContext.getArray();
		ArrayList<String> vocab=_rin.getSortedWordListString();
		String contextFile =_opt.outputDir+"contextDict"+_opt.algorithm+_opt.typeofDecomp;
		int counter=0;
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(contextFile),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		if(_opt.typeofDecomp.equals("WvsL")||_opt.typeofDecomp.equals("WvsR")||_opt.typeofDecomp.equals("2viewWvsL")||_opt.typeofDecomp.equals("2viewWvsR"))
			counter=_opt.contextSizeOneSide;
		else
			counter=2*_opt.contextSizeOneSide;
		int c=0;
		
		double maxInThisDimension= 0;
		for (int i=0; i<counter*(vocab.size()+1); i++) {
			double arr=0;
			maxInThisDimension= 0;
			if(i%(vocab.size()+1)==0 && i!=0)
				c++;
				
			if (i==0 || i==c*(vocab.size()+1)){
				writer.write("<OOV>");
				writer.write(' ');
			}
			else{
				if(i<=vocab.size()){
					writer.write(vocab.get(i-1));
					writer.write(' ');
				}else{
					writer.write(vocab.get(i-(c*vocab.size())-1-c));
					writer.write(' ');
				}
					
			}
			for(int k=0;k<_opt.hiddenStateSize;k++){
				arr=eigenDictArrContext[i][k];
				if(maxInThisDimension<Math.abs(arr))
					maxInThisDimension=Math.abs(arr);
			}
			
			for (int j=0; j<_opt.hiddenStateSize;j++){
				if(i<=vocab.size()){
					if ( j != _opt.hiddenStateSize-1){
						writer.write(Double.toString(eigenDictArrContext[i][j]/maxInThisDimension));
						writer.write(' ');
					}
					else{
						writer.write(Double.toString(eigenDictArrContext[i][j]/maxInThisDimension));
						writer.write('\n');
					}
				}
				else{
					if ( j != _opt.hiddenStateSize-1){
						writer.write(Double.toString(eigenDictArrContext[i-c*vocab.size()][j]/maxInThisDimension));
						writer.write(' ');
					}
					else{
						writer.write(Double.toString(eigenDictArrContext[i-c*vocab.size()][j]/maxInThisDimension));
						writer.write('\n');
					}
				}
			}
		}
		writer.close();
	}

	
	public void writeEigContextVectorsRandom() throws IOException{
		Matrix eigenDictContext=(Matrix)_matrices[1];
		double[][] eigenDictArrContext=eigenDictContext.getArray();
		ArrayList<String> vocab=_rin.getSortedWordListString();
		String contextFile =_opt.outputDir+"contextDictRandom";
		
		Random r= new Random();
		int counter=0;
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(contextFile),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		if(_opt.typeofDecomp.equals("WvsL")||_opt.typeofDecomp.equals("WvsR")||_opt.typeofDecomp.equals("2viewWvsL")||_opt.typeofDecomp.equals("2viewWvsR"))
			counter=_opt.contextSizeOneSide;
		else
			counter=2*_opt.contextSizeOneSide;
		int c=0;
		for (int i=0; i<counter*(vocab.size()+1); i++) {
			if(i%(vocab.size()+1)==0 && i!=0)
				c++;
				
			if (i==0 || i==c*(vocab.size()+1)){
				writer.write("<OOV>");
				writer.write(' ');
			}
			else{
				if(i<=vocab.size()){
					writer.write(vocab.get(i-1));
					writer.write(' ');
				}else{
					writer.write(vocab.get(i-(c*vocab.size())-1-c));
					writer.write(' ');
				}
					
			}
			for (int j=0; j<_opt.hiddenStateSize;j++){
				
					if ( j != _opt.hiddenStateSize-1){
						writer.write(Double.toString(r.nextGaussian()));
						writer.write(' ');
					}
					else{
						writer.write(Double.toString(r.nextGaussian()));
						writer.write('\n');
					}
		
			}
		}
		writer.close();
	}
	
	
	
	public void writeContextSpecificEmbed(Matrix contextSpecificEmbed) throws IOException {
		int i=0,idxDoc=0,idx=0;
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.contextSpecificEmbed),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	
	while(idxDoc<_allDocs.size()){	
			int tok_idx=0;	
			ArrayList<Integer> doc=_allDocs.get(idxDoc++);
			
			double maxInThisDimension= 0;
			while(tok_idx<doc.size()){
				double arr=0;
				maxInThisDimension= 0;
				writer.write(_rin.getTokForIntTrain(idx++));
				writer.write(' ');
				
				for(int k=0;k<_opt.hiddenStateSize*2;k++){
					arr=contextSpecificEmbed.get(i, k);
					if(maxInThisDimension<Math.abs(arr))
						maxInThisDimension=Math.abs(arr);
				}
				
				for (int j=0;j<_opt.hiddenStateSize*2;j++){
					
					
					
					
					if ( j != (2*_opt.hiddenStateSize)-1){
						writer.write(Float.toString((float) (contextSpecificEmbed.get(i, j)/maxInThisDimension)));
						writer.write(' ');
					}
					else{
						writer.write(Float.toString((float) (contextSpecificEmbed.get(i, j)/maxInThisDimension)));
						writer.write('\n');
					}
				}
				i++;
				tok_idx++;
			}
	}
		writer.close();
		
	}
	
	
	

	
	public void writeContextSpecificEmbedLRMVL(Matrix contextSpecificEmbed) throws IOException {
		int i=0,idxDoc=0,idx=0;
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.contextSpecificEmbed),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	
	while(idxDoc<_allDocs.size()){	
			int tok_idx=0;	
			ArrayList<Integer> doc=_allDocs.get(idxDoc++);
			
			double maxInThisDimension= 0;
			while(tok_idx<doc.size()){
				double arr=0;
				maxInThisDimension=0;
				writer.write(_rin.getTokForIntTrain(idx++));
				writer.write(' ');
				
				for(int k=0;k<_opt.hiddenStateSize*3;k++){
					arr=contextSpecificEmbed.get(i, k);
					if(maxInThisDimension<Math.abs(arr))
						maxInThisDimension=Math.abs(arr);
				}
				
				for (int j=0;j<_opt.hiddenStateSize*3;j++){
					
					if ( j != (3*_opt.hiddenStateSize)-1){
						writer.write(Double.toString(contextSpecificEmbed.get(i, j)/maxInThisDimension));
						writer.write(' ');
					}
					else{
						writer.write(Double.toString(contextSpecificEmbed.get(i, j)/maxInThisDimension));
						writer.write('\n');
					}
				}
				i++;
				tok_idx++;
			}
	}
		writer.close();
		
	}

	

	
	public void writeEigenDict() throws IOException{
		DenseDoubleMatrix2D dictLMatrix=createDenseMatrixCOLT((Matrix)_matrices[0]);
		double[][] dictL=dictLMatrix.toArray();
		ArrayList<String> vocab=_rin.getSortedWordListString();
		String eigenDict=_opt.outputDir+"eigenDict"+_opt.algorithm+_opt.typeofDecomp;
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(eigenDict),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		double maxInThisDimension= 0;
		for (int i=0; i<=vocab.size(); i++) {
			double arr=0;
			maxInThisDimension= 0;
			if (i==0){
				writer.write("<OOV>");
				writer.write(' ');
			}
			else{
				writer.write(vocab.get(i-1));
				writer.write(' ');
			}
			
			for(int k=0;k<_opt.hiddenStateSize;k++){
				arr=dictL[i][k];
				if(maxInThisDimension<Math.abs(arr))
					maxInThisDimension=Math.abs(arr);
			}
			
			for (int j=0; j<_opt.hiddenStateSize;j++){
				
				if ( j != _opt.hiddenStateSize-1){
					writer.write(Double.toString(dictL[i][j]/maxInThisDimension));
					writer.write(' ');
				}
				else{
					writer.write(Double.toString(dictL[i][j]/maxInThisDimension));
					writer.write('\n');
				}
			}
		}
		writer.close();
	}

	
	
	public void writeEigenDictCPCA() throws IOException{
		DenseDoubleMatrix2D dictLMatrix=createDenseMatrixCOLT((Matrix)_matrices[0]);
		double[][] dictL=dictLMatrix.toArray();
		ArrayList<String> vocab=_rin.getSortedWordListString();
		String eigenDict=_opt.outputDir+"eigenDict"+_opt.algorithm+_opt.typeofDecomp;
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(eigenDict),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		double[] maxInThisDimension= new double[vocab.size()+1];
		for (int i=0; i<=vocab.size(); i++) {
			double arr=0;
			if (i==0){
				writer.write("<OOV>");
				writer.write(' ');
			}
			else{
				writer.write(vocab.get(i-1));
				writer.write(' ');
			}
			
			
			for (int j=0; j<_opt.hiddenStateSize;j++){
				
				if ( j != _opt.hiddenStateSize-1){
					writer.write(Double.toString(dictL[i][j]));
					writer.write(' ');
				}
				else{
					writer.write(Double.toString(dictL[i][j]));
					writer.write('\n');
				}
			}	
		}
		writer.close();
	}
	
	
	public void writeEigenDictRandom() throws IOException{
		Random r= new Random();
		DenseDoubleMatrix2D dictLMatrix=createDenseMatrixCOLT((Matrix)_matrices[0]);
		double[][] dictL=dictLMatrix.toArray();
		ArrayList<String> vocab=_rin.getSortedWordListString();
		String eigenDict=_opt.outputDir+"eigenDictRandom";
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(eigenDict),"UTF8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		for (int i=0; i<=vocab.size(); i++) {
			
			if (i==0){
				writer.write("<OOV>");
				writer.write(' ');
			}
			else{
				writer.write(vocab.get(i-1));
				writer.write(' ');
			}
			for (int j=0; j<_opt.hiddenStateSize;j++){
				
				if ( j != _opt.hiddenStateSize-1){
					writer.write(Double.toString(r.nextGaussian()));
					writer.write(' ');
				}
				else{
					writer.write(Double.toString(r.nextGaussian()));
					writer.write('\n');
				}
			}
			
		}
		writer.close();
	}

	
	public void writeContextObliviousEmbedNewData(Matrix embedMatrix,
			HashMap<String, Integer> wordDict) throws IOException {
		
		BufferedWriter writer1=null;
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(_opt.contextOblEmbed),"UTF8"));
			writer1=new BufferedWriter(new OutputStreamWriter(new FileOutputStream("Output_Files/newembeds"),"UTF8"));
			
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	ArrayList<ArrayList<String>> _allDocs=new ArrayList<ArrayList<String>>();
	int idxDoc=0;
	_allDocs = readTrainData();
	Random r =new Random();
			for(String keys: wordDict.keySet()){
				int val=wordDict.get(keys);
				writer1.write(keys);
				writer1.write(' ');
				for (int j=embedMatrix.getColumnDimension()-1;j>=0;j--){
					
					double newVal =embedMatrix.get(val, j)+ r.nextGaussian()*0.01;
					if ( j != 0){
						writer1.write(Double.toString(newVal));
						writer1.write(' ');
					}
					else{
						writer1.write(Double.toString(newVal));
						writer1.write('\n');
						}
				}
			}
	writer1.close();
	
	while(idxDoc<_allDocs.size()){	
			int tok_idx=0;	
			ArrayList<String> doc=_allDocs.get(idxDoc++);
			
			while(tok_idx<doc.size()){
				writer.write(doc.get(tok_idx));
				writer.write(' ');
				for (int j=0;j<embedMatrix.getColumnDimension();j++){
					
					if ( j != (embedMatrix.getColumnDimension())-1){
						if(wordDict.get(doc.get(tok_idx))!=null)
							writer.write(Double.toString(embedMatrix.get(wordDict.get(doc.get(tok_idx)), j)));
						else
							writer.write(Double.toString(embedMatrix.get(wordDict.get(("<OOV>")), j)));
						writer.write(' ');
					}
					else{
						if(wordDict.get(doc.get(tok_idx))!=null)
							writer.write(Double.toString(embedMatrix.get(wordDict.get(doc.get(tok_idx)), j)));
						else
							writer.write(Double.toString(embedMatrix.get(wordDict.get(("<OOV>")), j)));
						writer.write('\n');
					}
				}
			
				tok_idx++;
			}
	}
		writer.close();	
		
	}
	
	public ArrayList<ArrayList<String>> readTrainData() throws IOException{
	
	BufferedReader	in=new BufferedReader(new InputStreamReader(new FileInputStream(_opt.trainfile), "UTF8"));			
	 String docEndSymbol="DOCSTART-X-0";
	ArrayList<ArrayList<String>> allDocs=new ArrayList<ArrayList<String>>();	
	ArrayList<String> eachDoc=new ArrayList<String>();
	
	String line=in.readLine();
	while (line != null ) {
		
		
		if(line.equals("")){
			line=in.readLine();
			continue;
		}
			
		if (!line.equals(docEndSymbol)){
			ArrayList<String> norm1=new ArrayList<String>();
			
			norm1=tokenize(line);
			for(String w:norm1){
				eachDoc.add(w);
			}
			
		}
		else{
			allDocs.add((ArrayList<String>) eachDoc.clone());
			
			eachDoc.clear();
		}
		line=in.readLine();
	}
	    in.close();
	   allDocs.add((ArrayList<String>) eachDoc.clone());
	   
	   return allDocs;
}
	
	public static ArrayList<String> tokenize(String s){
		if(s==null)
			return null;
		ArrayList<String> res=new ArrayList<String>();
		StringTokenizer st=new StringTokenizer(s," ");
		while(st.hasMoreTokens())
			res.add(st.nextToken());
		return res;
	}
	public void writeSparseMatrix(FlexCompRowMatrix wtlrMatrix,FlexCompRowMatrix wtwMatrix) {
		
		DenseDoubleMatrix2D d =new DenseDoubleMatrix2D(wtlrMatrix.numRows(),wtlrMatrix.numColumns());
		
		for(MatrixEntry e: wtlrMatrix){
			d.setQuick(e.row(), e.column(), e.get()/wtwMatrix.get(e.row(), e.row()));
		}
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream("ContextMatrix"),"UTF8"));
		} catch (UnsupportedEncodingException e2) {
			// TODO Auto-generated catch block
			e2.printStackTrace();
		} catch (FileNotFoundException e2) {
			// TODO Auto-generated catch block
			e2.printStackTrace();
		}
		
		try {
		for(int i=0; i<wtlrMatrix.numRows(); i++){
			for(int j=0; j<wtlrMatrix.numColumns(); j++){

				if(j!=wtlrMatrix.numColumns()-1){
					writer.write(Double.toString(d.getQuick(i, j)));
					writer.write(' ');	
				}
				else{
					writer.write(Double.toString(d.getQuick(i, j)));
				}
			}
			writer.write('\n');
		}
		writer.close();
		} 
		catch (IOException e1) {
			e1.printStackTrace();
		}
		//wtlrMatrix.
		
		//writer.write(Double.toString(embedMatrix.get(wordDict.get(("<OOV>")), j)));
		//writer.write('\n');
		
		
		for (MatrixEntry e : wtlrMatrix){
			d.setQuick(e.row(), e.column(), (int)e.get());
		}
		
	
		
	}	
	
public void writeDenseMatrix(DenseDoubleMatrix2D y) {
		
		try {
			writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream("Matrix"),"UTF8"));
		} catch (UnsupportedEncodingException e2) {
			// TODO Auto-generated catch block
			e2.printStackTrace();
		} catch (FileNotFoundException e2) {
			// TODO Auto-generated catch block
			e2.printStackTrace();
		}
		
		try {
		for(int i=0; i<y.rows(); i++){
			for(int j=0; j<y.columns(); j++){

				if(j!=y.columns()-1){
					writer.write(Double.toString(y.getQuick(i, j)));
					writer.write(' ');	
				}
				else{
					writer.write(Double.toString(y.getQuick(i, j)));
				}
			}
			writer.write('\n');
		}
		writer.close();
		} 
		catch (IOException e1) {
			e1.printStackTrace();
		}
	
	}	
}
