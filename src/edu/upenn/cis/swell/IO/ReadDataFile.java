package edu.upenn.cis.swell.IO;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.Map.Entry;

import edu.upenn.cis.swell.MathUtils.CenterScaleNormalizeUtils;


public class ReadDataFile implements Serializable {
	private static boolean lowercase=false;
	private static boolean normalize=false;
	private BufferedReader  in = null;
	Options _opt;
	static final long serialVersionUID = 42L;
	HashMap<Integer,String> strMap=null;
	private static String docEndSymbol="DOCSTART-X-0";
	ArrayList<Integer> docSizes=new ArrayList<Integer>();
	long numTokens =0;
	
	private HashMap<String,Integer> countW=new HashMap<String,Integer>();
	private HashMap<String,Integer> countContext=new HashMap<String,Integer>();
	private HashMap<String,Integer> corpusIntMapped =new HashMap<String,Integer>();
	private HashMap<String,Integer> corpusIntMappedContext =new HashMap<String,Integer>();
	private ValueComparator bvc=new ValueComparator(countW);
	private TreeMap<String,Integer> sorted_countW = new TreeMap(bvc);
	private ValueComparator bvcContext=new ValueComparator(countContext);
	private TreeMap<String,Integer> sorted_countContext = new TreeMap(bvcContext);
	
	ArrayList<ArrayList<Integer>> allDocs=new ArrayList<ArrayList<Integer>>();
	
    Object[] wordHashMaps;
	
	
	
	public ReadDataFile(Options options){
		try{
			_opt=options;
			normalize=_opt.normalize;
			lowercase=_opt.lowercase;
			docEndSymbol=_opt.docSeparator;
	
			
			
		}catch(Exception e){
			e.printStackTrace();
			System.exit(0);
		}
	}
	
	@SuppressWarnings("unchecked")
	public void readAllDocs(int dataOption) throws Exception{
		
		if (dataOption==0)
			in= new BufferedReader(new FileReader(_opt.unlabDataTrainfile));
		
		if (dataOption==1){
			in= new BufferedReader(new FileReader(_opt.trainfile));
			strMap=new HashMap<Integer,String>();
		}
		
		
		
		ArrayList<Integer> eachDoc=new ArrayList<Integer>();
		
		String line=in.readLine();
		int docCounter=0,numDocs=0;
		int idx=0;
		while (line != null ) {
			
			
			if(line.equals("")){
				line=in.readLine();
				continue;
			}
				
			
			if (!line.equals(docEndSymbol)){
				docCounter++;
				ArrayList<String> norm1=new ArrayList<String>();
				
				norm1=tokenize(line);
				for(String w:norm1){
					if (dataOption==1){
						strMap.put(idx++, w);
					}
				}
				
				if (lowercase)
					norm1=lowercase(norm1);
				if (normalize)
					norm1=(normalize(norm1));
				
				
				for(String w:norm1){
					
					int wInt=-1;
					if (corpusIntMapped.containsKey(w))
						wInt=corpusIntMapped.get(w);
					else
						wInt=corpusIntMapped.get("<OOV>");
					numTokens++;
					eachDoc.add(wInt);
				}
				
			}
			else{
				numDocs++;
				
				allDocs.add((ArrayList<Integer>) eachDoc.clone());
				
				docSizes.add(docCounter);
				docCounter=0;
				eachDoc.clear();
			}
			line=in.readLine();
		}
		    in.close();
		   allDocs.add((ArrayList<Integer>) eachDoc.clone());
		    
		    docSizes.add(docCounter);

		
	}
	
	
	
	
	public void readAllDocsNGrams() throws Exception{
		
		
		in= new BufferedReader(new FileReader(_opt.unlabDataTrainfile));
		Object[] wordHashMaps =new Object[1];
		HashMap<Double,Double> word_contextCounts =new HashMap<Double,Double>();
		CenterScaleNormalizeUtils utils=new CenterScaleNormalizeUtils(_opt);
		
		String line=in.readLine();
		while (line != null ) {
			
			
			if(line.equals("")){
				line=in.readLine();
				continue;
			}
				
				ArrayList<String> norm1=new ArrayList<String>();
				
				norm1=tokenize(line);
				
				
				if (lowercase)
					norm1=lowercase(norm1);
				if (normalize)
					norm1=(normalize(norm1));
				
				Iterator<String> itArrList=norm1.iterator();
				
				while (itArrList.hasNext()){
					int wInt=-1,cInt=-1;
					String w=itArrList.next();//The first entry is the word, then the context, then the counts.
					String c=itArrList.next();//Context
					if (corpusIntMapped.containsKey(w))
						wInt=corpusIntMapped.get(w);
					else
						wInt=corpusIntMapped.get("<OOV>");
					
					if (corpusIntMappedContext.containsKey(c))
						cInt=corpusIntMappedContext.get(c);
					else
						cInt=corpusIntMappedContext.get("<OOV>");
					numTokens++;
					word_contextCounts.put(utils.cantorPairingMap(wInt,cInt), Double.parseDouble(itArrList.next()));
				}
				
			line=in.readLine();
		}
		    in.close();
		   
		    wordHashMaps[0]= word_contextCounts;
		    
		
	}
	
	
public void readAllDocsNGramsSingleVocab() throws Exception{
		
		
		in= new BufferedReader(new FileReader(_opt.unlabDataTrainfile));
		HashMap<Double,Double> word_contextCounts,word_contextCounts3WR1,word_contextCounts3WR2,
		word_contextCounts3WL1,word_contextCounts3WL2,word_contextCounts3WL,word_contextCounts3WR,word_contextCounts3R1R2_OR_LR_OR_L1L2;
		
		
			word_contextCounts =new HashMap<Double, Double>();
	
			word_contextCounts3WR1 =new HashMap<Double, Double>();
			word_contextCounts3WR2 =new HashMap<Double, Double>();
	
			word_contextCounts3R1R2_OR_LR_OR_L1L2 =new HashMap<Double, Double>();
			
			word_contextCounts3WL1 =new HashMap<Double, Double>();
			word_contextCounts3WL2 =new HashMap<Double, Double>();
		
			word_contextCounts3WL =new HashMap<Double, Double>();
			word_contextCounts3WR =new HashMap<Double, Double>();
			
			
			HashMap<Double, Double> word_contextCounts5WR1 = new HashMap<Double, Double>();
			HashMap<Double, Double> word_contextCounts5WR2 = new HashMap<Double, Double>();
			HashMap<Double, Double> word_contextCounts5WR3 = new HashMap<Double, Double>();
			HashMap<Double, Double> word_contextCounts5WR4 = new HashMap<Double, Double>();
	
			HashMap<Double, Double> word_contextCounts5WL1 = new HashMap<Double, Double>();
			HashMap<Double, Double> word_contextCounts5WL2 = new HashMap<Double, Double>();
			HashMap<Double, Double> word_contextCounts5WL3 = new HashMap<Double, Double>();
			HashMap<Double, Double> word_contextCounts5WL4 = new HashMap<Double, Double>();
			
			HashMap<Double, Double> word_contextCounts5R1R2_OR_L1L2_OR_L1R1 = new HashMap<Double, Double>();
			HashMap<Double, Double> word_contextCounts5R1R3_OR_L1L3_OR_L1R2 = new HashMap<Double, Double>();
			HashMap<Double, Double> word_contextCounts5R1R4_OR_L1L4_OR_L2R1 = new HashMap<Double, Double>();
			HashMap<Double, Double> word_contextCounts5R2R3_OR_L2L3_OR_L2R2 = new HashMap<Double, Double>();
			HashMap<Double, Double> word_contextCounts5R2R4_OR_L2L4_OR_L1L2 = new HashMap<Double, Double>();
			HashMap<Double, Double> word_contextCounts5R3R4_OR_L3L4_OR_R1R2 = new HashMap<Double, Double>();
				
			
		
		CenterScaleNormalizeUtils utils=new CenterScaleNormalizeUtils(_opt);
		
		 wordHashMaps=new Object[1];
		    if(_opt.numGrams==3)
		    	wordHashMaps=new Object[3];
		    if(_opt.numGrams==5)
		    	wordHashMaps=new Object[10];
		
		String line=in.readLine();
		while (line != null ) {
			
			
			if(line.equals("")){
				line=in.readLine();
				continue;
			}
				
				ArrayList<String> norm1=new ArrayList<String>();
				
				norm1=tokenize(line);
				
				
				if (lowercase)
					norm1=lowercase(norm1);
				if (normalize)
					norm1=(normalize(norm1));
				
				Iterator<String> itArrList=norm1.iterator();
				
				if(_opt.numGrams==2){
					while (itArrList.hasNext()){
						int wInt=-1,cInt=-1;
						String w=itArrList.next();//The first entry is the word, then the context, then the counts.
						String c=itArrList.next();//Context
						if (corpusIntMapped.containsKey(w))
							wInt=corpusIntMapped.get(w);
						else
							wInt=corpusIntMapped.get("<OOV>");
						if (corpusIntMapped.containsKey(c))
							cInt=corpusIntMapped.get(c);
						else
							cInt=corpusIntMapped.get("<OOV>");
						numTokens++;
						double existC =0;
						if(word_contextCounts.get(utils.cantorPairingMap(wInt,cInt))!=null)
							existC = word_contextCounts.get(utils.cantorPairingMap(wInt,cInt));
						word_contextCounts.put(utils.cantorPairingMap(wInt,cInt), existC+Double.parseDouble(itArrList.next()));
						}
				}
				
				
				
				/////
				if(_opt.numGrams==3 && (_opt.typeofDecomp.equals("2viewWvsL") || _opt.typeofDecomp.equals("WvsL") )){
					while (itArrList.hasNext()){
						int wInt=-1,l2Int=-1,l1Int=-1;
						String l2=itArrList.next();//The first entry is the left context 2 etc.
						String l1=itArrList.next();
						String w=itArrList.next();
						if (corpusIntMapped.containsKey(w))
							wInt=corpusIntMapped.get(w);
						else
							wInt=corpusIntMapped.get("<OOV>");
						if (corpusIntMapped.containsKey(l2))
							l2Int=corpusIntMapped.get(l2);
						else
							l2Int=corpusIntMapped.get("<OOV>");
						if (corpusIntMapped.containsKey(l1))
							l1Int=corpusIntMapped.get(l1);
						else
							l1Int=corpusIntMapped.get("<OOV>");
						numTokens++;
						double l1ExistingCounts=0,l2ExistingCounts=0,l1l2ExistingCounts=0;
						if(word_contextCounts3WR1.get(utils.cantorPairingMap(wInt,l1Int))!=null)
							l1ExistingCounts=word_contextCounts3WR1.get(utils.cantorPairingMap(wInt,l1Int));
						if(word_contextCounts3WR2.get(utils.cantorPairingMap(wInt,l2Int))!=null)
							l2ExistingCounts=word_contextCounts3WR2.get(utils.cantorPairingMap(wInt,l2Int));
						if(word_contextCounts3R1R2_OR_LR_OR_L1L2.get(utils.cantorPairingMap(l1Int,l2Int))!=null)
							l1l2ExistingCounts=word_contextCounts3R1R2_OR_LR_OR_L1L2.get(utils.cantorPairingMap(l1Int,l2Int));
						
						double count=Double.parseDouble(itArrList.next());
						word_contextCounts3R1R2_OR_LR_OR_L1L2.put(utils.cantorPairingMap(l1Int,l2Int), l1l2ExistingCounts +count);
						word_contextCounts3WL1.put(utils.cantorPairingMap(wInt,l1Int), l1ExistingCounts +count);
						word_contextCounts3WL2.put(utils.cantorPairingMap(wInt,l2Int), l2ExistingCounts + count);
						}
				}
				////////
				if(_opt.numGrams==3 && (_opt.typeofDecomp.equals("2viewWvsR") || _opt.typeofDecomp.equals("WvsR") )){
					while (itArrList.hasNext()){
						int wInt=-1,r2Int=-1,r1Int=-1;
						String w=itArrList.next();//The first entry is the word w.
						String r1=itArrList.next();
						String r2=itArrList.next();
						if (corpusIntMapped.containsKey(w))
							wInt=corpusIntMapped.get(w);
						else
							wInt=corpusIntMapped.get("<OOV>");
						if (corpusIntMapped.containsKey(r2))
							r2Int=corpusIntMapped.get(r2);
						else
							r2Int=corpusIntMapped.get("<OOV>");
						if (corpusIntMapped.containsKey(r1))
							r1Int=corpusIntMapped.get(r1);
						else
							r1Int=corpusIntMapped.get("<OOV>");
						numTokens++;
						double r1ExistingCounts=0,r2ExistingCounts=0,r1r2ExistingCounts=0;
						if(word_contextCounts3WR1.get(utils.cantorPairingMap(wInt,r1Int))!=null)
							r1ExistingCounts=word_contextCounts3WR1.get(utils.cantorPairingMap(wInt,r1Int));
						if(word_contextCounts3WR2.get(utils.cantorPairingMap(wInt,r2Int))!=null)
							r2ExistingCounts=word_contextCounts3WR2.get(utils.cantorPairingMap(wInt,r2Int));
						if(word_contextCounts3R1R2_OR_LR_OR_L1L2.get(utils.cantorPairingMap(r1Int,r2Int))!=null)
							r1r2ExistingCounts=word_contextCounts3R1R2_OR_LR_OR_L1L2.get(utils.cantorPairingMap(r1Int,r2Int));
						
						double count=Double.parseDouble(itArrList.next());
						word_contextCounts3R1R2_OR_LR_OR_L1L2.put(utils.cantorPairingMap(r1Int,r2Int), r1r2ExistingCounts +count);
						word_contextCounts3WR1.put(utils.cantorPairingMap(wInt,r1Int), r1ExistingCounts +count);
						word_contextCounts3WR2.put(utils.cantorPairingMap(wInt,r2Int), r2ExistingCounts + count);
						}
				}
				///////
				if(_opt.numGrams==3 && (_opt.typeofDecomp.equals("2viewWvsLR") || _opt.typeofDecomp.equals("WvsLR") || _opt.typeofDecomp.equals("TwoStepLRvsW"))){
					while (itArrList.hasNext()){
						int wInt=-1,rInt=-1,lInt=-1;
						String l=itArrList.next();//The first entry is the left context 2 etc.
						String r=itArrList.next();
						String w=itArrList.next();
						if (corpusIntMapped.containsKey(w))
							wInt=corpusIntMapped.get(w);
						else
							wInt=corpusIntMapped.get("<OOV>");
						if (corpusIntMapped.containsKey(l))
							lInt=corpusIntMapped.get(l);
						else
							lInt=corpusIntMapped.get("<OOV>");
						if (corpusIntMapped.containsKey(r))
							rInt=corpusIntMapped.get(r);
						else
							rInt=corpusIntMapped.get("<OOV>");
						numTokens++;
						
						double wrExistingCounts=0,wlExistingCounts=0,lrExistingCounts=0;
						
						if(word_contextCounts3WR.get(utils.cantorPairingMap(wInt,rInt)) !=null)
							wrExistingCounts=word_contextCounts3WR.get(utils.cantorPairingMap(wInt,rInt));
						
						if(word_contextCounts3WL.get(utils.cantorPairingMap(wInt,lInt))!=null)
							wlExistingCounts=word_contextCounts3WL.get(utils.cantorPairingMap(wInt,lInt));
						
						if(word_contextCounts3R1R2_OR_LR_OR_L1L2.get(utils.cantorPairingMap(lInt,rInt))!=null)
							lrExistingCounts=word_contextCounts3R1R2_OR_LR_OR_L1L2.get(utils.cantorPairingMap(lInt,rInt));
						
						
						double count=Double.parseDouble(itArrList.next());
						word_contextCounts3R1R2_OR_LR_OR_L1L2.put(utils.cantorPairingMap(lInt,rInt), lrExistingCounts +count);
						word_contextCounts3WL.put(utils.cantorPairingMap(wInt,lInt), wlExistingCounts +count);
						word_contextCounts3WR.put(utils.cantorPairingMap(wInt,rInt), wrExistingCounts + count);
						}
				}
			/////
				if(_opt.numGrams==5 &&(_opt.typeofDecomp.equals("2viewWvsL") || _opt.typeofDecomp.equals("WvsL") )){
					while (itArrList.hasNext()){
						int wInt=-1,l2Int=-1,l1Int=-1,l3Int=-1,l4Int=-1;
						String l4=itArrList.next();//The first entry is the left context 4 etc.
						String l3=itArrList.next();
						String l2=itArrList.next();
						String l1=itArrList.next();
						String w=itArrList.next();
						if (corpusIntMapped.containsKey(w))
							wInt=corpusIntMapped.get(w);
						else
							wInt=corpusIntMapped.get("<OOV>");
						if (corpusIntMapped.containsKey(l4))
							l4Int=corpusIntMapped.get(l4);
						else
							l4Int=corpusIntMapped.get("<OOV>");
						
						if (corpusIntMapped.containsKey(l3))
							l3Int=corpusIntMapped.get(l3);
						else
							l3Int=corpusIntMapped.get("<OOV>");
						
						if (corpusIntMapped.containsKey(l2))
							l2Int=corpusIntMapped.get(l2);
						else
							l2Int=corpusIntMapped.get("<OOV>");
						
						if (corpusIntMapped.containsKey(l1))
							l1Int=corpusIntMapped.get(l1);
						else
							l1Int=corpusIntMapped.get("<OOV>");
						
						numTokens++;
						
						double l1ExistingCounts=0,l2ExistingCounts=0,l3ExistingCounts=0,l4ExistingCounts=0,
								l1l2ExistingCounts=0,l1l3ExistingCounts=0,l1l4ExistingCounts=0,l2l3ExistingCounts=0,
										l2l4ExistingCounts=0,l3l4ExistingCounts=0;
						
						if(word_contextCounts5WL1.get(utils.cantorPairingMap(wInt,l1Int)) !=null)
							l1ExistingCounts=word_contextCounts5WL1.get(utils.cantorPairingMap(wInt,l1Int));
						
						if(word_contextCounts5WL2.get(utils.cantorPairingMap(wInt,l2Int)) !=null)
							l2ExistingCounts=word_contextCounts5WL2.get(utils.cantorPairingMap(wInt,l2Int));
						
						if(word_contextCounts5WL3.get(utils.cantorPairingMap(wInt,l3Int)) !=null)
							l3ExistingCounts=word_contextCounts5WL3.get(utils.cantorPairingMap(wInt,l3Int));
						
						if(word_contextCounts5WL4.get(utils.cantorPairingMap(wInt,l4Int)) !=null)
							l4ExistingCounts=word_contextCounts5WL4.get(utils.cantorPairingMap(wInt,l4Int));
						
						
						
						if(word_contextCounts5R1R2_OR_L1L2_OR_L1R1.get(utils.cantorPairingMap(l1Int,l2Int)) !=null)
							l1l2ExistingCounts=word_contextCounts5R1R2_OR_L1L2_OR_L1R1.get(utils.cantorPairingMap(l1Int,l2Int));
						
						if(word_contextCounts5R1R3_OR_L1L3_OR_L1R2.get(utils.cantorPairingMap(l1Int,l3Int)) !=null)
							l1l3ExistingCounts=word_contextCounts5R1R3_OR_L1L3_OR_L1R2.get(utils.cantorPairingMap(l1Int,l3Int));
						
						if(word_contextCounts5R1R4_OR_L1L4_OR_L2R1.get(utils.cantorPairingMap(l1Int,l4Int)) !=null)
							l1l4ExistingCounts=word_contextCounts5R1R4_OR_L1L4_OR_L2R1.get(utils.cantorPairingMap(l1Int,l4Int));
						
						if(word_contextCounts5R2R3_OR_L2L3_OR_L2R2.get(utils.cantorPairingMap(l2Int,l3Int)) !=null)
							l2l3ExistingCounts=word_contextCounts5R2R3_OR_L2L3_OR_L2R2.get(utils.cantorPairingMap(l2Int,l3Int));
						
						if(word_contextCounts5R2R4_OR_L2L4_OR_L1L2.get(utils.cantorPairingMap(l2Int,l4Int)) !=null)
							l2l4ExistingCounts=word_contextCounts5R2R4_OR_L2L4_OR_L1L2.get(utils.cantorPairingMap(l2Int,l4Int));
						
						if(word_contextCounts5R3R4_OR_L3L4_OR_R1R2.get(utils.cantorPairingMap(l3Int,l4Int)) !=null)
							l3l4ExistingCounts=word_contextCounts5R3R4_OR_L3L4_OR_R1R2.get(utils.cantorPairingMap(l3Int,l4Int));
						
						
						double count=Double.parseDouble(itArrList.next());
						
						word_contextCounts5R1R2_OR_L1L2_OR_L1R1.put(utils.cantorPairingMap(l1Int,l2Int), l1l2ExistingCounts +count);
						word_contextCounts5R1R3_OR_L1L3_OR_L1R2.put(utils.cantorPairingMap(l1Int,l3Int), l1l3ExistingCounts +count);
						word_contextCounts5R1R4_OR_L1L4_OR_L2R1.put(utils.cantorPairingMap(l1Int,l4Int), l1l4ExistingCounts +count);
						word_contextCounts5R2R3_OR_L2L3_OR_L2R2.put(utils.cantorPairingMap(l2Int,l3Int), l2l3ExistingCounts +count);
						word_contextCounts5R2R4_OR_L2L4_OR_L1L2.put(utils.cantorPairingMap(l2Int,l4Int), l2l4ExistingCounts +count);
						word_contextCounts5R3R4_OR_L3L4_OR_R1R2.put(utils.cantorPairingMap(l3Int,l4Int), l3l4ExistingCounts +count);
						
						word_contextCounts5WL1.put(utils.cantorPairingMap(wInt,l1Int), l1ExistingCounts +count);
						word_contextCounts5WL2.put(utils.cantorPairingMap(wInt,l2Int), l2ExistingCounts +count);
						word_contextCounts5WL3.put(utils.cantorPairingMap(wInt,l3Int), l3ExistingCounts +count);
						word_contextCounts5WL4.put(utils.cantorPairingMap(wInt,l4Int), l4ExistingCounts +count);
					}
				}
				////////
				if(_opt.numGrams==5 && (_opt.typeofDecomp.equals("2viewWvsR") || _opt.typeofDecomp.equals("WvsR") )){
					while (itArrList.hasNext()){
						int wInt=-1,r2Int=-1,r1Int=-1,r3Int=-1,r4Int=-1;
						String w=itArrList.next();//The first entry is the word etc.
						String r1=itArrList.next();
						String r2=itArrList.next();
						String r3=itArrList.next();
						String r4=itArrList.next();
						if (corpusIntMapped.containsKey(w))
							wInt=corpusIntMapped.get(w);
						else
							wInt=corpusIntMapped.get("<OOV>");
						if (corpusIntMapped.containsKey(r4))
							r4Int=corpusIntMapped.get(r4);
						else
							r4Int=corpusIntMapped.get("<OOV>");
						
						if (corpusIntMapped.containsKey(r3))
							r3Int=corpusIntMapped.get(r3);
						else
							r3Int=corpusIntMapped.get("<OOV>");
						
						if (corpusIntMapped.containsKey(r2))
							r2Int=corpusIntMapped.get(r2);
						else
							r2Int=corpusIntMapped.get("<OOV>");
						
						if (corpusIntMapped.containsKey(r1))
							r1Int=corpusIntMapped.get(r1);
						else
							r1Int=corpusIntMapped.get("<OOV>");
						
						numTokens++;
						double r1ExistingCounts=0,r2ExistingCounts=0,r3ExistingCounts=0,r4ExistingCounts=0,
								r1r2ExistingCounts=0,r1r3ExistingCounts=0,r1r4ExistingCounts=0,r2r3ExistingCounts=0,
										r2r4ExistingCounts=0,r3r4ExistingCounts=0;
						
						if(word_contextCounts5WR1.get(utils.cantorPairingMap(wInt,r1Int)) !=null)
							r1ExistingCounts=word_contextCounts5WR1.get(utils.cantorPairingMap(wInt,r1Int));
						
						if(word_contextCounts5WR2.get(utils.cantorPairingMap(wInt,r2Int)) !=null)
							r2ExistingCounts=word_contextCounts5WR2.get(utils.cantorPairingMap(wInt,r2Int));
						
						if(word_contextCounts5WR3.get(utils.cantorPairingMap(wInt,r3Int)) !=null)
							r3ExistingCounts=word_contextCounts5WR3.get(utils.cantorPairingMap(wInt,r3Int));
						
						if(word_contextCounts5WR4.get(utils.cantorPairingMap(wInt,r4Int)) !=null)
							r4ExistingCounts=word_contextCounts5WR4.get(utils.cantorPairingMap(wInt,r4Int));
						
						if(word_contextCounts5R1R2_OR_L1L2_OR_L1R1.get(utils.cantorPairingMap(r1Int,r2Int)) !=null)
							r1r2ExistingCounts=word_contextCounts5R1R2_OR_L1L2_OR_L1R1.get(utils.cantorPairingMap(r1Int,r2Int));
						
						if(word_contextCounts5R1R3_OR_L1L3_OR_L1R2.get(utils.cantorPairingMap(r1Int,r3Int)) !=null)
							r1r3ExistingCounts=word_contextCounts5R1R3_OR_L1L3_OR_L1R2.get(utils.cantorPairingMap(r1Int,r3Int));
						
						if(word_contextCounts5R1R4_OR_L1L4_OR_L2R1.get(utils.cantorPairingMap(r1Int,r4Int)) !=null)
							r1r4ExistingCounts=word_contextCounts5R1R4_OR_L1L4_OR_L2R1.get(utils.cantorPairingMap(r1Int,r4Int));
						
						if(word_contextCounts5R2R3_OR_L2L3_OR_L2R2.get(utils.cantorPairingMap(r2Int,r3Int)) !=null)
							r2r3ExistingCounts=word_contextCounts5R2R3_OR_L2L3_OR_L2R2.get(utils.cantorPairingMap(r2Int,r3Int));
						
						if(word_contextCounts5R2R4_OR_L2L4_OR_L1L2.get(utils.cantorPairingMap(r2Int,r4Int)) !=null)
							r2r4ExistingCounts=word_contextCounts5R2R4_OR_L2L4_OR_L1L2.get(utils.cantorPairingMap(r2Int,r4Int));
						
						if(word_contextCounts5R3R4_OR_L3L4_OR_R1R2.get(utils.cantorPairingMap(r3Int,r4Int)) !=null)
							r3r4ExistingCounts=word_contextCounts5R3R4_OR_L3L4_OR_R1R2.get(utils.cantorPairingMap(r3Int,r4Int));
						
						
						double count=Double.parseDouble(itArrList.next());
						
						word_contextCounts5R1R2_OR_L1L2_OR_L1R1.put(utils.cantorPairingMap(r1Int,r2Int), r1r2ExistingCounts +count);
						word_contextCounts5R1R3_OR_L1L3_OR_L1R2.put(utils.cantorPairingMap(r1Int,r3Int), r1r3ExistingCounts +count);
						word_contextCounts5R1R4_OR_L1L4_OR_L2R1.put(utils.cantorPairingMap(r1Int,r4Int), r1r4ExistingCounts +count);
						word_contextCounts5R2R3_OR_L2L3_OR_L2R2.put(utils.cantorPairingMap(r2Int,r3Int), r2r3ExistingCounts +count);
						word_contextCounts5R2R4_OR_L2L4_OR_L1L2.put(utils.cantorPairingMap(r2Int,r4Int), r2r4ExistingCounts +count);
						word_contextCounts5R3R4_OR_L3L4_OR_R1R2.put(utils.cantorPairingMap(r3Int,r4Int), r3r4ExistingCounts +count);
						
						word_contextCounts5WR1.put(utils.cantorPairingMap(wInt,r1Int), r1ExistingCounts +count);
						word_contextCounts5WR2.put(utils.cantorPairingMap(wInt,r2Int), r2ExistingCounts +count);
						word_contextCounts5WR3.put(utils.cantorPairingMap(wInt,r3Int), r3ExistingCounts +count);
						word_contextCounts5WR4.put(utils.cantorPairingMap(wInt,r4Int), r4ExistingCounts +count);
					}
				}
				///////
				if(_opt.numGrams==5 && (_opt.typeofDecomp.equals("2viewWvsLR") || _opt.typeofDecomp.equals("WvsLR") || _opt.typeofDecomp.equals("TwoStepLRvsW"))){
					while (itArrList.hasNext()){
						int wInt=-1,l2Int=-1,l1Int=-1,r2Int=-1,r1Int=-1;
						String l2=itArrList.next();//The first entry is the left context 2 etc.
						String l1=itArrList.next();
						String w=itArrList.next();
						String r1=itArrList.next();
						String r2=itArrList.next();
						
						if (corpusIntMapped.containsKey(w))
							wInt=corpusIntMapped.get(w);
						else
							wInt=corpusIntMapped.get("<OOV>");
						if (corpusIntMapped.containsKey(r2))
							r2Int=corpusIntMapped.get(r2);
						else
							r2Int=corpusIntMapped.get("<OOV>");
						
						if (corpusIntMapped.containsKey(r1))
							r1Int=corpusIntMapped.get(r1);
						else
							r1Int=corpusIntMapped.get("<OOV>");
						
						if (corpusIntMapped.containsKey(l2))
							l2Int=corpusIntMapped.get(l2);
						else
							l2Int=corpusIntMapped.get("<OOV>");
						
						if (corpusIntMapped.containsKey(l1))
							l1Int=corpusIntMapped.get(l1);
						else
							l1Int=corpusIntMapped.get("<OOV>");
						
						numTokens++;
						double l1ExistingCounts=0,l2ExistingCounts=0,r1ExistingCounts=0,r2ExistingCounts=0,
								l1r1ExistingCounts=0,l1r2ExistingCounts=0,l2r1ExistingCounts=0,l2r2ExistingCounts=0,
								l1l2ExistingCounts=0,r1r2ExistingCounts=0;
						
						if(word_contextCounts5WL1.get(utils.cantorPairingMap(wInt,l1Int)) !=null)
							l1ExistingCounts=word_contextCounts5WL1.get(utils.cantorPairingMap(wInt,l1Int));
						
						if(word_contextCounts5WL2.get(utils.cantorPairingMap(wInt,l2Int)) !=null)
							l2ExistingCounts=word_contextCounts5WL2.get(utils.cantorPairingMap(wInt,l2Int));
						
						if(word_contextCounts5WR1.get(utils.cantorPairingMap(wInt,r1Int)) !=null)
							r1ExistingCounts=word_contextCounts5WR1.get(utils.cantorPairingMap(wInt,r1Int));
						
						if(word_contextCounts5WR2.get(utils.cantorPairingMap(wInt,r2Int)) !=null)
							r2ExistingCounts=word_contextCounts5WR2.get(utils.cantorPairingMap(wInt,r2Int));
						
						if(word_contextCounts5R1R2_OR_L1L2_OR_L1R1.get(utils.cantorPairingMap(l1Int,r1Int)) !=null)
							l1r1ExistingCounts=word_contextCounts5R1R2_OR_L1L2_OR_L1R1.get(utils.cantorPairingMap(l1Int,r1Int));
						
						if(word_contextCounts5R1R3_OR_L1L3_OR_L1R2.get(utils.cantorPairingMap(l1Int,r2Int)) !=null)
							l1r2ExistingCounts=word_contextCounts5R1R3_OR_L1L3_OR_L1R2.get(utils.cantorPairingMap(l1Int,r2Int));
						
						if(word_contextCounts5R1R4_OR_L1L4_OR_L2R1.get(utils.cantorPairingMap(l2Int,r1Int)) !=null)
							l2r1ExistingCounts=word_contextCounts5R1R4_OR_L1L4_OR_L2R1.get(utils.cantorPairingMap(l2Int,r1Int));
						
						if(word_contextCounts5R2R3_OR_L2L3_OR_L2R2.get(utils.cantorPairingMap(l2Int,r2Int)) !=null)
							l2r2ExistingCounts=word_contextCounts5R2R3_OR_L2L3_OR_L2R2.get(utils.cantorPairingMap(l2Int,r2Int));
						
						if(word_contextCounts5R2R4_OR_L2L4_OR_L1L2.get(utils.cantorPairingMap(l1Int,l2Int)) !=null)
							l1l2ExistingCounts=word_contextCounts5R2R4_OR_L2L4_OR_L1L2.get(utils.cantorPairingMap(l1Int,l2Int));
						
						if(word_contextCounts5R3R4_OR_L3L4_OR_R1R2.get(utils.cantorPairingMap(r1Int,r2Int)) !=null)
							r1r2ExistingCounts=word_contextCounts5R3R4_OR_L3L4_OR_R1R2.get(utils.cantorPairingMap(r1Int,r2Int));
						
						
						double count=Double.parseDouble(itArrList.next());
						
						word_contextCounts5R1R2_OR_L1L2_OR_L1R1.put(utils.cantorPairingMap(l1Int,r1Int), l1r1ExistingCounts +count);
						word_contextCounts5R1R3_OR_L1L3_OR_L1R2.put(utils.cantorPairingMap(l1Int,r2Int), l1r2ExistingCounts +count);
						word_contextCounts5R1R4_OR_L1L4_OR_L2R1.put(utils.cantorPairingMap(l2Int,r1Int), l2r1ExistingCounts +count);
						word_contextCounts5R2R3_OR_L2L3_OR_L2R2.put(utils.cantorPairingMap(l2Int,r2Int), l2r2ExistingCounts +count);
						word_contextCounts5R2R4_OR_L2L4_OR_L1L2.put(utils.cantorPairingMap(l1Int,l2Int), l1l2ExistingCounts +count);
						word_contextCounts5R3R4_OR_L3L4_OR_R1R2.put(utils.cantorPairingMap(r1Int,r2Int), r1r2ExistingCounts +count);
						
						
						word_contextCounts5WL1.put(utils.cantorPairingMap(wInt,l1Int), l1ExistingCounts +count);
						word_contextCounts5WL2.put(utils.cantorPairingMap(wInt,l2Int), l2ExistingCounts +count);
						word_contextCounts5WR1.put(utils.cantorPairingMap(wInt,r1Int), r1ExistingCounts +count);
						word_contextCounts5WR2.put(utils.cantorPairingMap(wInt,r2Int), r2ExistingCounts +count);					}
				}
			line=in.readLine();
		}
		    in.close();
		    
		if(_opt.numGrams==2)    
			wordHashMaps[0] =(Object) word_contextCounts; 
		
		if(_opt.numGrams==3 && (_opt.typeofDecomp.equals("2viewWvsL") || _opt.typeofDecomp.equals("WvsL")) ){    
			wordHashMaps[0] =(Object) word_contextCounts3WL1;
			wordHashMaps[1] =(Object) word_contextCounts3WL2;
			wordHashMaps[2] =(Object) word_contextCounts3R1R2_OR_LR_OR_L1L2;
		}
		
		if(_opt.numGrams==3 &&  (_opt.typeofDecomp.equals("2viewWvsR") || _opt.typeofDecomp.equals("WvsR")) ){   
			wordHashMaps[0] =(Object) word_contextCounts3WR1; 
			wordHashMaps[1] =(Object) word_contextCounts3WR2; 
			wordHashMaps[2] =(Object) word_contextCounts3R1R2_OR_LR_OR_L1L2;
		}
		
		if(_opt.numGrams==3 && (_opt.typeofDecomp.equals("2viewWvsLR")|| _opt.typeofDecomp.equals("WvsLR") || _opt.typeofDecomp.equals("TwoStepLRvsW")) ){    
			wordHashMaps[0] =(Object) word_contextCounts3WL; 
			wordHashMaps[1] =(Object) word_contextCounts3WR; 
			wordHashMaps[2] =(Object) word_contextCounts3R1R2_OR_LR_OR_L1L2;
		}
		////////////////
		
		if(_opt.numGrams==5 && (_opt.typeofDecomp.equals("2viewWvsL") || _opt.typeofDecomp.equals("WvsL")) ){    
			wordHashMaps[0] =(Object) word_contextCounts5WL1;
			wordHashMaps[1] =(Object) word_contextCounts5WL2;
			wordHashMaps[2] =(Object) word_contextCounts5WL3;
			wordHashMaps[3] =(Object) word_contextCounts5WL4;
			
			
			wordHashMaps[4] =(Object) word_contextCounts5R1R2_OR_L1L2_OR_L1R1;
			wordHashMaps[5] =(Object) word_contextCounts5R1R3_OR_L1L3_OR_L1R2;
			wordHashMaps[6] =(Object) word_contextCounts5R1R4_OR_L1L4_OR_L2R1;
			wordHashMaps[7] =(Object) word_contextCounts5R2R3_OR_L2L3_OR_L2R2;
			wordHashMaps[8] =(Object) word_contextCounts5R2R4_OR_L2L4_OR_L1L2;
			wordHashMaps[9] =(Object) word_contextCounts5R3R4_OR_L3L4_OR_R1R2;
		}
		
		if(_opt.numGrams==5 && (_opt.typeofDecomp.equals("2viewWvsR") || _opt.typeofDecomp.equals("WvsR")) ){   
			wordHashMaps[0] =(Object) word_contextCounts5WR1;
			wordHashMaps[1] =(Object) word_contextCounts5WR2;
			wordHashMaps[2] =(Object) word_contextCounts5WR3;
			wordHashMaps[3] =(Object) word_contextCounts5WR4;
			
			wordHashMaps[4] =(Object) word_contextCounts5R1R2_OR_L1L2_OR_L1R1;
			wordHashMaps[5] =(Object) word_contextCounts5R1R3_OR_L1L3_OR_L1R2;
			wordHashMaps[6] =(Object) word_contextCounts5R1R4_OR_L1L4_OR_L2R1;
			wordHashMaps[7] =(Object) word_contextCounts5R2R3_OR_L2L3_OR_L2R2;
			wordHashMaps[8] =(Object) word_contextCounts5R2R4_OR_L2L4_OR_L1L2;
			wordHashMaps[9] =(Object) word_contextCounts5R3R4_OR_L3L4_OR_R1R2;

		}
		
		if(_opt.numGrams==5 && (_opt.typeofDecomp.equals("2viewWvsLR")|| _opt.typeofDecomp.equals("WvsLR") || _opt.typeofDecomp.equals("TwoStepLRvsW")) ){    
			wordHashMaps[0] =(Object) word_contextCounts5WL1;
			wordHashMaps[1] =(Object) word_contextCounts5WL2;
			wordHashMaps[2] =(Object) word_contextCounts5WR1;
			wordHashMaps[3] =(Object) word_contextCounts5WR2;
			
			wordHashMaps[4] =(Object) word_contextCounts5R1R2_OR_L1L2_OR_L1R1;
			wordHashMaps[5] =(Object) word_contextCounts5R1R3_OR_L1L3_OR_L1R2;
			wordHashMaps[6] =(Object) word_contextCounts5R1R4_OR_L1L4_OR_L2R1;
			wordHashMaps[7] =(Object) word_contextCounts5R2R3_OR_L2L3_OR_L2R2;
			wordHashMaps[8] =(Object) word_contextCounts5R2R4_OR_L2L4_OR_L1L2;
			wordHashMaps[9] =(Object) word_contextCounts5R3R4_OR_L3L4_OR_R1R2;

		}
		    
		
	}
	
	public String getTokForIntTrain(int i){
		return strMap.get(i);
	}
	
	public HashMap<String, Integer> convertAllDocsIntNGrams() throws Exception {
		
		in= new BufferedReader(new FileReader(_opt.unlabDataTrainfile));
		
		
		String line=in.readLine();
		int docCounter=0,numDocs=0;
		while (line != null ) {
			
			if(line.equals("")){
				line=in.readLine();
				continue;
			}
				
			
			if (!line.equals(docEndSymbol)){
				docCounter++;
				int count=-1;
				ArrayList<String> norm1=new ArrayList<String>();
				norm1=tokenize(line);
				
				if (lowercase)
					norm1=lowercase(norm1);
				if (normalize)
					norm1=(normalize(norm1));
				for(String w:norm1){
					count++; 
					if (count==0){ //Add the actual word in leftmost position to the hash table.
					if (countW.containsKey(w))
						countW.put(w,countW.get(w)+1);
					else
						countW.put(w,1);
					}
					if (count==1 ){
						if (countContext.containsKey(w))
							countContext.put(w,countContext.get(w)+1);
						else
							countContext.put(w,1);
					}
				}
			}
			
			line=in.readLine();
		}
		    in.close();
		    
		    sorted_countW.putAll(countW);
		    sorted_countContext.putAll(countContext);
			int i=0,j=0;
			if (i==0){
				corpusIntMapped.put("<OOV>",i++);
				
			}
			if (j==0){
				corpusIntMappedContext.put("<OOV>",j++);
			}
			
			for (String keys:sorted_countW.keySet()){
				if (i<= _opt.vocabSize)
					corpusIntMapped.put(keys,i++);
			}
			
			for (String keys:sorted_countContext.keySet()){
				if (j<= _opt.numLabels*_opt.vocabSize)
					corpusIntMappedContext.put(keys,j++);
			}
		    
		return corpusIntMapped;
		
	}
	
public HashMap<String, Integer> convertAllDocsIntNGramsSingleVocab() throws Exception {
		
		in= new BufferedReader(new FileReader(_opt.unlabDataTrainfile));
		
		
		String line=in.readLine();
		int docCounter=0,numDocs=0;
		while (line != null ) {
			
			if(line.equals("")){
				line=in.readLine();
				continue;
			}
				
			
			if (!line.equals(docEndSymbol)){
				docCounter++;
				int count=-1;
				ArrayList<String> norm1=new ArrayList<String>();
				norm1=tokenize(line);
				
				if (lowercase)
					norm1=lowercase(norm1);
				if (normalize)
					norm1=(normalize(norm1));
				for(String w:norm1){
					count++; 
				if(count==0 || count==1){
					
					if (countW.containsKey(w))
						countW.put(w,countW.get(w)+1);
					else
						countW.put(w,1);
				}
				}
			}
			
			line=in.readLine();
		}
		    in.close();
		    
		    sorted_countW.putAll(countW);
		   // sorted_countContext.putAll(countContext);
			int i=0,j=0;
			if (i==0){
				corpusIntMapped.put("<OOV>",i++);
				
			}
			
			for (String keys:sorted_countW.keySet()){
				if (i<= _opt.vocabSize)
					corpusIntMapped.put(keys,i++);
			}
			
			
		return corpusIntMapped;
		
	}
	
	public HashMap<String,Integer> convertAllDocsInt(int dataOption) throws Exception{
		
		if (dataOption==0)
			in= new BufferedReader(new FileReader(_opt.unlabDataTrainfile));
		
		if (dataOption==1)
			in= new BufferedReader(new FileReader(_opt.trainfile));
		
		
		String line=in.readLine();
		int docCounter=0,numDocs=0;
		while (line != null ) {
			
			if(line.equals("")){
				line=in.readLine();
				continue;
			}
				
			
			if (!line.equals(docEndSymbol)){
				docCounter++;
				ArrayList<String> norm1=new ArrayList<String>();
				norm1=tokenize(line);
				if (lowercase)
					norm1=lowercase(norm1);
				if (normalize)
					norm1=(normalize(norm1));
				for(String w:norm1){
					if (countW.containsKey(w))
						countW.put(w,countW.get(w)+1);
					else
						countW.put(w,1);
				}
			}
			
			line=in.readLine();
		}
		    in.close();
		    
		    sorted_countW.putAll(countW);
			int i=0;
			if (i==0)
				corpusIntMapped.put("<OOV>",i++);
			
			for (String keys:sorted_countW.keySet()){
				if (i<= _opt.vocabSize)
					corpusIntMapped.put(keys,i++);
			}
		    
		return corpusIntMapped;
		
	}
	
	public Object[] getAllDocsNGrams(){
		return wordHashMaps;
	}
	
	public ArrayList<ArrayList<Integer>> getAllDocs(){
		return allDocs;
	}
	
	public  HashMap<String,Integer> getCorpusIntMapped(){
		return corpusIntMapped;
	}
	
	public  HashMap<String,Integer> getCorpusContextIntMapped(){
		return corpusIntMappedContext;
	}
	
	public  String getTokForInt(int i){
		
		String _str=null;
		Set<Entry<String,Integer>> keyvals=corpusIntMapped.entrySet();
		Iterator<Entry<String,Integer>> valIter=keyvals.iterator();
		while(valIter.hasNext()){
			Entry<String,Integer> entry=valIter.next();
			if(entry.getValue()==i)
				return entry.getKey();
		}
		return _str;
		
	}
	
	public ArrayList<Integer> getSortedCountList(){
		ArrayList<Integer> countsList= new ArrayList<Integer>();
		for (Integer keys:sorted_countW.values()){
			countsList.add(keys);
		}
		return countsList;
	}
	
	public ArrayList<Integer> getSortedWordList(){
		ArrayList<Integer> wordsList= new ArrayList<Integer>();
		int count=0;
		
		for (Integer keys:corpusIntMapped.values()){
			if (count <_opt.vocabSize){
				wordsList.add(keys);
				count++;
			}
			else 
				break;
			
		}
		return wordsList;
	}
	
	public ArrayList<String> getSortedWordListString(){
		ArrayList<String> wordsList= new ArrayList<String>();
		int count=0;
		
		for (String keys:sorted_countW.keySet()){
			if (count <_opt.vocabSize){
				wordsList.add(keys);
				count++;
			}
			else 
				break;
			
		}
		return wordsList;
	}
	
	public ArrayList<String> getSortedWordListContextString(){
		ArrayList<String> wordsList= new ArrayList<String>();
		int count=0;
		
		for (String keys:sorted_countContext.keySet()){
			if (count <_opt.numLabels*_opt.vocabSize){
				wordsList.add(keys);
				count++;
			}
			else 
				break;
			
		}
		return wordsList;
	}
	
	/*public void updateDocsWithInts(HashMap<Integer, Document> hMapDocs, HashMap<String,Integer> corpus_IntMapped) {
		int tok;
		int idx=0;
		while(idx<hMapDocs.size()){
				int idxTok=0;
				Document doc=hMapDocs.get(idx++);

				while(idxTok<doc.size()){
					tok=doc.get(idxTok++);
					if (corpus_IntMapped.containsValue(tok))
						tok.setTokenIntId(corpus_IntMapped.get(tok.getToken()));
					else{
						tok.setTokenIntId(corpus_IntMapped.get("<OOV>"));
						//tok.setisOOV(true);
					}
				}
			}
		
		//updateLRContexts(hMapDocs);
	} */

	
	
	/*public void CreateIntMapping(){
		Iterator<ArrayList<ArrayList<String>>> it_AllWordsDocs= _allWords.iterator();
		
		while(it_AllWordsDocs.hasNext()){
			Iterator<ArrayList<String>> it_AllWords=it_AllWordsDocs.next().iterator();
			while(it_AllWords.hasNext()){
				ArrayList<String> docWords=it_AllWords.next();
				Iterator<String> it_Words= docWords.iterator();
				while(it_Words.hasNext()){
					String w=it_Words.next();
					if (countW.containsKey(w))
						countW.put(w,countW.get(w)+1);
					else
						countW.put(w,1);
				}
			}
	}
		sorted_countW.putAll(countW);
		int i=0;
		if (i==0)
			corpusIntMapped.put("<OOV>",i++);
		
		for (String keys:sorted_countW.keySet()){
			if (i<= _vocabSize)
				corpusIntMapped.put(keys,i++);
		}
	} */
	
	protected ArrayList<String> normalize (ArrayList<String> s) {
		ArrayList<String> norm=new ArrayList<String>();
		Iterator<String> itNorm=s.iterator();
		while(itNorm.hasNext())
		{
			String s1=itNorm.next();
			if(s1.matches("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+"))
				norm.add("<num>");
			else
				norm.add(s1);
		}

		return (ArrayList<String>) norm.clone();
}
	
	
	public void serializeCorpusIntMapped(){
		File f= new File(_opt.serializeCorpus);
		
		try{
			ObjectOutput corpus=new ObjectOutputStream(new FileOutputStream(f));
			corpus.writeObject((Object)this.corpusIntMapped);
			corpus.flush( );
			corpus.close( );
			
			
			System.out.println("=======Serialized the Corpus (Int Mapped)=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
	}
	
	
	public long getNumTokens(){
		return numTokens;
	}
	
	public void serializeCorpusIntMappedContext(){
		File f= new File(_opt.serializeCorpus+"Contexts");
		
		try{
			ObjectOutput corpus=new ObjectOutputStream(new FileOutputStream(f));
			corpus.writeObject((Object)this.corpusIntMappedContext);
			corpus.flush( );
			corpus.close( );
			
			
			System.out.println("=======Serialized the Corpus Contexts (Int Mapped)=======");
		}
		catch (IOException ioe){
			System.out.println(ioe.getMessage());
		}
	}
	
	protected ArrayList<String> lowercase (ArrayList<String> s) {
		ArrayList<String> lower=new ArrayList<String>();
		Iterator<String> itlower=s.iterator();
		while(itlower.hasNext())
		    lower.add(itlower.next().toLowerCase().trim());
		
		return (ArrayList<String>) lower.clone();
}
	
	public ArrayList<Integer> getDocSizes(){
		return docSizes;
	}

	public void setCorpusIntMapped(HashMap<String,Integer> corpusIntMapped){
		this.corpusIntMapped=corpusIntMapped;
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
	public void close(){
		try{
			this.in.close();
		}catch(Exception E){}
	}
	
	class ValueComparator implements Comparator,Serializable {

		  Map base;
		  public ValueComparator(Map base) {
		      this.base = base;
		  }

		  public int compare(Object a, Object b) {

		    if((Integer)base.get(a) < (Integer)base.get(b) ||(Integer)base.get(a) == (Integer)base.get(b)) {
		      return 1;
		    }  else {
		      return -1;
		    }
		  }
	}

	

	
		

}
