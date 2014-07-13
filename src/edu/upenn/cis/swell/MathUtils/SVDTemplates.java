package edu.upenn.cis.swell.MathUtils;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */


import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.io.UnsupportedEncodingException;

import Jama.Matrix;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.QR;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.algo.decomposition.DenseDoubleSingularValueDecomposition;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import edu.upenn.cis.swell.IO.ContextPCAWriter;
import edu.upenn.cis.swell.IO.Options;

public class SVDTemplates implements Serializable {

	Options _opt;
	int _dimension2=0;
	private DenseDoubleMatrix2D dictMatrixCOLT=null;
	long startTime=0,endTime=0;
	static final long serialVersionUID = 42L;
	double[] sVals;
	
	public SVDTemplates(Options opt, int numDocs){
		this(opt);
		_dimension2=numDocs;
		
	}
	
	public SVDTemplates(Options opt){
		_opt=opt;
		
	}
	
	public SparseDoubleMatrix2D computeSparseInverse(SparseDoubleMatrix2D X){
		
		//Computes inverse of diagonally dominant sparse matrices by power series expansion as
		// (A+ eB)^-1 = A^-1 - e. A^-1.B.A^-1 + e^2.A^-1.B.A^-1.B.A^-1...Below we only use the first two terms.
		//In our case A is a diagonal matrix so its inverse is easy and B is the remainder. 
		
		
		
		//SparseDoubleMatrix2D tempMat=new SparseDoubleMatrix2D(X.rows(),X.columns(),0,0.7,0.75);
		//SparseDoubleMatrix2D auxMat=new SparseDoubleMatrix2D(X.rows(),X.columns(),0,0.7,0.75);
		SparseDoubleMatrix2D diagInvEntries=new SparseDoubleMatrix2D(X.rows(),X.columns(),0,0.7,0.75);
		//SparseDoubleMatrix2D OffdiagEntries=new SparseDoubleMatrix2D(X.rows(),X.columns(),0,0.7,0.75);
		
		
		System.out.println("++Beginning Sparse Inverse++");
		
		
		for(int i=0; i<X.rows();i++){ 
			
			if(X.get(i, i)!=0){
				if(!_opt.sqRootNorm)
					diagInvEntries.set(i, i, 1/X.get(i, i));
				else
					diagInvEntries.set(i, i, 1/Math.sqrt(X.get(i, i)));
			}else{
				diagInvEntries.set(i, i, 10000);//Some large value
			}
		}
		System.out.println("++Finished Sparse Inverse++");
		
		/*
		if(!_opt.diagOnlyInverse){
		
			for(int i=0; i<X.rows();i++){ 
				for(int j=0; j<X.columns();j++){ 
					if(i!=j){
						OffdiagEntries.set(i, j, X.get(i, j));
							}
												}
								}
			
			diagInvEntries.zMult(OffdiagEntries, tempMat);
			tempMat.zMult(diagInvEntries, auxMat);
	
			auxMat.assign(diagInvEntries, DoublePlusMultFirst.minusMult(1));
			return auxMat;
		}
		else{
		*/
		
			return diagInvEntries;
		//}
		
	}
	
	
public SparseDoubleMatrix2D computeSparseInverseSqRoot(SparseDoubleMatrix2D X){
		
		SparseDoubleMatrix2D diagInvEntries=new SparseDoubleMatrix2D(X.rows(),X.columns(),0,0.7,0.75);
		
		
		System.out.println("++Beginning Sparse Inverse++");
		
		
		for(int i=0; i<X.rows();i++){ 
			
			if(X.get(i, i)!=0){
				diagInvEntries.set(i, i, 1/Math.sqrt(X.get(i, i)));
			}else{
				diagInvEntries.set(i, i, 10000);//Some large value
			}
		}
		System.out.println("++Finished Sparse Inverse Sq. Root++");
		
			return diagInvEntries;
		
	}
	
	
public FlexCompRowMatrix computeSparseInverseSqRoot(FlexCompRowMatrix X){
		
		FlexCompRowMatrix diagInvEntries=new FlexCompRowMatrix(X.numRows(),X.numColumns());
		
		
		System.out.println("++Beginning Sparse Inverse Sq. Root++");
		
		
		for(MatrixEntry e: X){
			if(e.row()==e.column() && e.get()!=0){
					diagInvEntries.set(e.row(),e.column() , 1/Math.sqrt(e.get()));
				
			}
			if(e.row()==e.column() && e.get()==0){
				diagInvEntries.set(e.row(),e.column() , 10000); //Some large value 
				
			}
			
		}
		
		
		System.out.println("++Finished Sparse Inverse Sq. Root++");
		
			return diagInvEntries;

		
	}
	
	
	
public FlexCompRowMatrix computeSparseInverse(FlexCompRowMatrix X){
		
		//Computes inverse of diagonally dominant sparse matrices by power series expansion as
		// (A+ eB)^-1 = A^-1 - e. A^-1.B.A^-1 + e^2.A^-1.B.A^-1.B.A^-1...Below we only use the first two terms.
		//In our case A is a diagonal matrix so its inverse is easy and B is the remainder. 
		
		
		
		//SparseDoubleMatrix2D tempMat=new SparseDoubleMatrix2D(X.rows(),X.columns(),0,0.7,0.75);
		//SparseDoubleMatrix2D auxMat=new SparseDoubleMatrix2D(X.rows(),X.columns(),0,0.7,0.75);
		FlexCompRowMatrix diagInvEntries=new FlexCompRowMatrix(X.numRows(),X.numColumns());
		//SparseDoubleMatrix2D OffdiagEntries=new SparseDoubleMatrix2D(X.rows(),X.columns(),0,0.7,0.75);
		
		
		System.out.println("++Beginning Sparse Inverse++");
		
		
		for(MatrixEntry e: X){
			if(e.row()==e.column() && e.get()!=0){
				if(!_opt.sqRootNorm)
					diagInvEntries.set(e.row(),e.column() , 1/e.get());
				else
					diagInvEntries.set(e.row(),e.column() , 1/Math.sqrt(e.get()));
				
			}
			if(e.row()==e.column() && e.get()==0){
				diagInvEntries.set(e.row(),e.column() , 10000); //Some large value 
				
			}
			
		}
		
		
		System.out.println("++Finished Sparse Inverse++");
		
		/*
		if(!_opt.diagOnlyInverse){
		
			for(int i=0; i<X.rows();i++){ 
				for(int j=0; j<X.columns();j++){ 
					if(i!=j){
						OffdiagEntries.set(i, j, X.get(i, j));
							}
												}
								}
			
			diagInvEntries.zMult(OffdiagEntries, tempMat);
			tempMat.zMult(diagInvEntries, auxMat);
	
			auxMat.assign(diagInvEntries, DoublePlusMultFirst.minusMult(1));
			return auxMat;
		}
		else{
		*/
		
			return diagInvEntries;
		//}
		
	}
	
	public Matrix computeSVD_Tropp(SparseDoubleMatrix2D X, DenseDoubleMatrix2D omega)
	{
		dictMatrixCOLT=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D Xomega=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize+20);//Oversample the required rank.
		DenseDoubleMatrix2D UhatTemp=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D UhatTemp1=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		
		DenseDoubleMatrix2D Uhat=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize);
		
		SparseDoubleMatrix2D sValsOmega=new SparseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D b=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_dimension2);
		DenseDoubleMatrix2D q=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize+20);

		for(int powIter =0; powIter <5;powIter++){
			startTime = System.currentTimeMillis();
			X.zMult(omega, Xomega);
			endTime = System.currentTimeMillis();
		//System.out.println("===Time taken for Multiplication: "+(endTime-startTime)/1000.0/60.0+" mins===");

			startTime = System.currentTimeMillis();

			QR qr=new QR(Xomega.rows(),Xomega.columns());


			DenseMatrix XomegaMTJ=MatrixFormatConversion.createDenseMatrixMTJ(Xomega);
			DenseMatrix qMTJ=qr.factor(XomegaMTJ).getQ();
			q=MatrixFormatConversion.createDenseMatrixCOLT(qMTJ);

			DenseDoubleAlgebra dalg=new DenseDoubleAlgebra();
			DenseDoubleMatrix2D qt=(DenseDoubleMatrix2D) dalg.transpose(q);

			qt.zMult(X, b);
			omega= (DenseDoubleMatrix2D)dalg.transpose(b);
			endTime = System.currentTimeMillis();
		}
		//System.out.println("===Time taken for QR and multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		startTime = System.currentTimeMillis();

		DenseDoubleSingularValueDecomposition svd= new DenseDoubleSingularValueDecomposition(b,true,false);
		
		
		UhatTemp1=(DenseDoubleMatrix2D) svd.getU();
		
		if(_opt.scaleBySingVals){
			sValsOmega=(SparseDoubleMatrix2D) svd.getS();
			sValsOmega.normalize();
			UhatTemp1.zMult(sValsOmega, UhatTemp);
		}
		else{
			UhatTemp=UhatTemp1;
		}
		
		
		for(int i=0; i<_opt.hiddenStateSize+20;i++){ //Take only the top k elements of the matrix after svd.
			for (int j=0;j<_opt.hiddenStateSize;j++){
				Uhat.set(i, j, UhatTemp.get(i, j));
			}
		}
		
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for SVD: "+(endTime-startTime)/1000.0/60.0+" mins===");




		startTime = System.currentTimeMillis();
		q.zMult(Uhat, dictMatrixCOLT);
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Final Multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		return MatrixFormatConversion.createDenseMatrixJAMA(dictMatrixCOLT);
	
}
	
	
	public Matrix computeSVD_Tropp(SparseDoubleMatrix2D X, DenseDoubleMatrix2D omega, int _dim2)
	{
		
/*		
		BufferedWriter writer=null;
		try {
			writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("Output_Files_new/Xmatrix"),"UTF8"));
		} catch (UnsupportedEncodingException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		try {
		for(int i=0; i < X.rows();i++ ){
			for(int j=0; j < X.columns();j++ ){
				
					writer.write(Double.toString(X.get(i, j)));
					writer.write(' ');
						
			}
			writer.write('\n');
		}
		writer.close();	
	}
	catch (IOException e) {
		e.printStackTrace();
		}	
		
*/		
		System.out.println("==Entering SVD==");
		
		dictMatrixCOLT=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D Xomega=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize+20);//Oversample the required rank.
		DenseDoubleMatrix2D UhatTemp=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D UhatTemp1=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		
		DenseDoubleMatrix2D Uhat=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize);
		
		System.out.println("==After UHat==");
		
		SparseDoubleMatrix2D sValsOmega=new SparseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D b=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_dim2);
		DenseDoubleMatrix2D q=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize+20);

		System.out.println("====Starting Power Iteration====");
		for(int powIter =0; powIter <5;powIter++){
			startTime = System.currentTimeMillis();
			X.zMult(omega, Xomega);
			endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Multiplication: "+(endTime-startTime)/1000.0/60.0+" mins===");

			startTime = System.currentTimeMillis();

			QR qr=new QR(Xomega.rows(),Xomega.columns());


			DenseMatrix XomegaMTJ=MatrixFormatConversion.createDenseMatrixMTJ(Xomega);
			DenseMatrix qMTJ=qr.factor(XomegaMTJ).getQ();
			q=MatrixFormatConversion.createDenseMatrixCOLT(qMTJ);

			DenseDoubleAlgebra dalg=new DenseDoubleAlgebra();
			DenseDoubleMatrix2D qt=(DenseDoubleMatrix2D) dalg.transpose(q);

			qt.zMult(X, b);
			omega= (DenseDoubleMatrix2D)dalg.transpose(b);
			endTime = System.currentTimeMillis();
		}
		System.out.println("===Time taken for QR and multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		startTime = System.currentTimeMillis();

		DenseDoubleSingularValueDecomposition svd= new DenseDoubleSingularValueDecomposition(b,true,false);
		
		
		UhatTemp1=(DenseDoubleMatrix2D) svd.getU();
		
		setSingularVals(svd.getSingularValues());
		
		if(_opt.scaleBySingVals){
			sValsOmega=(SparseDoubleMatrix2D) svd.getS();
			sValsOmega.normalize();
			UhatTemp1.zMult(sValsOmega, UhatTemp);
		}
		else{
			UhatTemp=UhatTemp1;
		}
		
		
		for(int i=0; i<_opt.hiddenStateSize+20;i++){ //Take only the top k elements of the matrix after svd.
			for (int j=0;j<_opt.hiddenStateSize;j++){
				Uhat.set(i, j, UhatTemp.get(i, j));
			}
		}
		
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for SVD: "+(endTime-startTime)/1000.0/60.0+" mins===");




		startTime = System.currentTimeMillis();
		q.zMult(Uhat, dictMatrixCOLT);
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Final Multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		return MatrixFormatConversion.createDenseMatrixJAMA(dictMatrixCOLT);
	
}
	
	
	public Matrix computeSVD_Tropp_1Stage(SparseDoubleMatrix2D X, DenseDoubleMatrix2D omega, int _dim2)
	{
		
		System.out.println("==Entering SVD==");
		
		dictMatrixCOLT=new DenseDoubleMatrix2D(X.rows(),2*_opt.hiddenStateSize);
		DenseDoubleMatrix2D Xomega=new DenseDoubleMatrix2D(X.rows(),2*_opt.hiddenStateSize+20);//Oversample the required rank.
		DenseDoubleMatrix2D UhatTemp=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize+20,2*_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D UhatTemp1=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize+20,2*_opt.hiddenStateSize+20);
		
		DenseDoubleMatrix2D Uhat=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize+20,2*_opt.hiddenStateSize);
		
		System.out.println("==After UHat==");
		
		SparseDoubleMatrix2D sValsOmega=new SparseDoubleMatrix2D(2*_opt.hiddenStateSize+20,2*_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D b=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize+20,_dim2);
		DenseDoubleMatrix2D q=new DenseDoubleMatrix2D(X.rows(),2*_opt.hiddenStateSize+20);

		System.out.println("====Starting Power Iteration====");
		for(int powIter =0; powIter <5;powIter++){
			startTime = System.currentTimeMillis();
			X.zMult(omega, Xomega);
			endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Multiplication: "+(endTime-startTime)/1000.0/60.0+" mins===");

			startTime = System.currentTimeMillis();

			QR qr=new QR(Xomega.rows(),Xomega.columns());


			DenseMatrix XomegaMTJ=MatrixFormatConversion.createDenseMatrixMTJ(Xomega);
			DenseMatrix qMTJ=qr.factor(XomegaMTJ).getQ();
			q=MatrixFormatConversion.createDenseMatrixCOLT(qMTJ);

			DenseDoubleAlgebra dalg=new DenseDoubleAlgebra();
			DenseDoubleMatrix2D qt=(DenseDoubleMatrix2D) dalg.transpose(q);

			qt.zMult(X, b);
			omega= (DenseDoubleMatrix2D)dalg.transpose(b);
			endTime = System.currentTimeMillis();
		}
		System.out.println("===Time taken for QR and multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		startTime = System.currentTimeMillis();

		DenseDoubleSingularValueDecomposition svd= new DenseDoubleSingularValueDecomposition(b,true,false);
		
		
		UhatTemp1=(DenseDoubleMatrix2D) svd.getU();
		
		setSingularVals(svd.getSingularValues());
		
		if(_opt.scaleBySingVals){
			sValsOmega=(SparseDoubleMatrix2D) svd.getS();
			sValsOmega.normalize();
			UhatTemp1.zMult(sValsOmega, UhatTemp);
		}
		else{
			UhatTemp=UhatTemp1;
		}
		
		
		for(int i=0; i<_opt.hiddenStateSize+20;i++){ //Take only the top k elements of the matrix after svd.
			for (int j=0;j<_opt.hiddenStateSize;j++){
				Uhat.set(i, j, UhatTemp.get(i, j));
			}
		}
		
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for SVD: "+(endTime-startTime)/1000.0/60.0+" mins===");




		startTime = System.currentTimeMillis();
		q.zMult(Uhat, dictMatrixCOLT);
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Final Multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		return MatrixFormatConversion.createDenseMatrixJAMA(dictMatrixCOLT);
	
}

	
	public Matrix computeSVD_Tropp(DenseDoubleMatrix2D X, DenseDoubleMatrix2D omega, int _dim2)
	{
		
	
		dictMatrixCOLT=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize);
		DenseDoubleMatrix2D Xomega=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize+20);//Oversample the required rank.
		DenseDoubleMatrix2D UhatTemp=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D UhatTemp1=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		
		DenseDoubleMatrix2D Uhat=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize);
		
		SparseDoubleMatrix2D sValsOmega=new SparseDoubleMatrix2D(_opt.hiddenStateSize+20,_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D b=new DenseDoubleMatrix2D(_opt.hiddenStateSize+20,_dim2);
		DenseDoubleMatrix2D q=new DenseDoubleMatrix2D(X.rows(),_opt.hiddenStateSize+20);

		System.out.println("====Starting Power Iteration====");
		for(int powIter =0; powIter <5;powIter++){
			startTime = System.currentTimeMillis();
			X.zMult(omega, Xomega);
			endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Multiplication: "+(endTime-startTime)/1000.0/60.0+" mins===");

			startTime = System.currentTimeMillis();

			QR qr=new QR(Xomega.rows(),Xomega.columns());


			DenseMatrix XomegaMTJ=MatrixFormatConversion.createDenseMatrixMTJ(Xomega);
			DenseMatrix qMTJ=qr.factor(XomegaMTJ).getQ();
			q=MatrixFormatConversion.createDenseMatrixCOLT(qMTJ);

			DenseDoubleAlgebra dalg=new DenseDoubleAlgebra();
			DenseDoubleMatrix2D qt=(DenseDoubleMatrix2D) dalg.transpose(q);

			qt.zMult(X, b);
			omega= (DenseDoubleMatrix2D)dalg.transpose(b);
			endTime = System.currentTimeMillis();
		}
		System.out.println("===Time taken for QR and multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		startTime = System.currentTimeMillis();

		DenseDoubleSingularValueDecomposition svd= new DenseDoubleSingularValueDecomposition(b,true,false);
		
		
		UhatTemp1=(DenseDoubleMatrix2D) svd.getU();
		
		setSingularVals(svd.getSingularValues());
		
		if(_opt.scaleBySingVals){
			sValsOmega=(SparseDoubleMatrix2D) svd.getS();
			sValsOmega.normalize();
			UhatTemp1.zMult(sValsOmega, UhatTemp);
		}
		else{
			UhatTemp=UhatTemp1;
		}
		
		
		for(int i=0; i<_opt.hiddenStateSize+20;i++){ //Take only the top k elements of the matrix after svd.
			for (int j=0;j<_opt.hiddenStateSize;j++){
				Uhat.set(i, j, UhatTemp.get(i, j));
			}
		}
		
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for SVD: "+(endTime-startTime)/1000.0/60.0+" mins===");




		startTime = System.currentTimeMillis();
		q.zMult(Uhat, dictMatrixCOLT);
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Final Multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		return MatrixFormatConversion.createDenseMatrixJAMA(dictMatrixCOLT);
	
}
	
	
	public Matrix computeSVD_Tropp_1Stage(DenseDoubleMatrix2D X, DenseDoubleMatrix2D omega, int _dim2)
	{
		
	
		dictMatrixCOLT=new DenseDoubleMatrix2D(X.rows(),2*_opt.hiddenStateSize);
		DenseDoubleMatrix2D Xomega=new DenseDoubleMatrix2D(X.rows(),2*_opt.hiddenStateSize+20);//Oversample the required rank.
		DenseDoubleMatrix2D UhatTemp=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize+20,2*_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D UhatTemp1=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize+20,2*_opt.hiddenStateSize+20);
		
		DenseDoubleMatrix2D Uhat=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize+20,2*_opt.hiddenStateSize);
		
		SparseDoubleMatrix2D sValsOmega=new SparseDoubleMatrix2D(2*_opt.hiddenStateSize+20,2*_opt.hiddenStateSize+20);
		DenseDoubleMatrix2D b=new DenseDoubleMatrix2D(2*_opt.hiddenStateSize+20,_dim2);
		DenseDoubleMatrix2D q=new DenseDoubleMatrix2D(X.rows(),2*_opt.hiddenStateSize+20);

		System.out.println("====Starting Power Iteration====");
		for(int powIter =0; powIter <5;powIter++){
			startTime = System.currentTimeMillis();
			X.zMult(omega, Xomega);
			endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Multiplication: "+(endTime-startTime)/1000.0/60.0+" mins===");

			startTime = System.currentTimeMillis();

			QR qr=new QR(Xomega.rows(),Xomega.columns());


			DenseMatrix XomegaMTJ=MatrixFormatConversion.createDenseMatrixMTJ(Xomega);
			DenseMatrix qMTJ=qr.factor(XomegaMTJ).getQ();
			q=MatrixFormatConversion.createDenseMatrixCOLT(qMTJ);

			DenseDoubleAlgebra dalg=new DenseDoubleAlgebra();
			DenseDoubleMatrix2D qt=(DenseDoubleMatrix2D) dalg.transpose(q);

			qt.zMult(X, b);
			omega= (DenseDoubleMatrix2D)dalg.transpose(b);
			endTime = System.currentTimeMillis();
		}
		System.out.println("===Time taken for QR and multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		startTime = System.currentTimeMillis();

		DenseDoubleSingularValueDecomposition svd= new DenseDoubleSingularValueDecomposition(b,true,false);
		
		
		UhatTemp1=(DenseDoubleMatrix2D) svd.getU();
		
		setSingularVals(svd.getSingularValues());
		
		if(_opt.scaleBySingVals){
			sValsOmega=(SparseDoubleMatrix2D) svd.getS();
			sValsOmega.normalize();
			UhatTemp1.zMult(sValsOmega, UhatTemp);
		}
		else{
			UhatTemp=UhatTemp1;
		}
		
		
		for(int i=0; i<2*_opt.hiddenStateSize+20;i++){ //Take only the top k elements of the matrix after svd.
			for (int j=0;j<2*_opt.hiddenStateSize;j++){
				Uhat.set(i, j, UhatTemp.get(i, j));
			}
		}
		
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for SVD: "+(endTime-startTime)/1000.0/60.0+" mins===");




		startTime = System.currentTimeMillis();
		q.zMult(Uhat, dictMatrixCOLT);
		endTime = System.currentTimeMillis();
		System.out.println("===Time taken for Final Multiply: "+(endTime-startTime)/1000.0/60.0+" mins===");

		return MatrixFormatConversion.createDenseMatrixJAMA(dictMatrixCOLT);
	
}

	private void setSingularVals(double[] singularValues) {
		this.sVals=singularValues;
		
	}

	public double[] getSingularVals(){
		return sVals;
	}

	public DenseDoubleMatrix2D computeDenseInverseSqRoot(DenseDoubleMatrix2D yty) {
			
		
		//ContextPCAWriter cw =new ContextPCAWriter(_opt);
		//cw.writeDenseMatrix(yty);
		
		DenseDoubleSingularValueDecomposition svd= new DenseDoubleSingularValueDecomposition(yty,true,true);
		DenseDoubleMatrix2D u=(DenseDoubleMatrix2D)svd.getU();
		SparseDoubleMatrix2D s =(SparseDoubleMatrix2D)svd.getS();
		SparseDoubleMatrix2D sinvSqRoot=new SparseDoubleMatrix2D(s.rows(),s.columns());
		DenseDoubleMatrix2D v=(DenseDoubleMatrix2D)svd.getV();
		DenseDoubleMatrix2D us =new DenseDoubleMatrix2D(u.rows(),s.columns());
		DenseDoubleMatrix2D x =new DenseDoubleMatrix2D(yty.rows(),yty.columns());
				
		
		for(int i=0; i< s.columns();i++){
			if(s.get(i,i)!=0)
				sinvSqRoot.set(i, i,1/Math.sqrt(s.get(i, i)));
			else
				sinvSqRoot.set(i, i,100000);
		}
		
		u.zMult(sinvSqRoot, us);
		us.zMult(v, x);
		return x;
	}
	
	public DenseDoubleMatrix2D computeDenseInverse(DenseDoubleMatrix2D yty) {
		
		DenseDoubleSingularValueDecomposition svd= new DenseDoubleSingularValueDecomposition(yty,true,true);
		DenseDoubleMatrix2D u=(DenseDoubleMatrix2D)svd.getU();
		SparseDoubleMatrix2D s =(SparseDoubleMatrix2D)svd.getS();
		SparseDoubleMatrix2D sinvSqRoot=new SparseDoubleMatrix2D(s.rows(),s.columns());
		DenseDoubleMatrix2D v=(DenseDoubleMatrix2D)svd.getV();
		DenseDoubleMatrix2D us =new DenseDoubleMatrix2D(u.rows(),s.columns());
		DenseDoubleMatrix2D x =new DenseDoubleMatrix2D(yty.rows(),yty.columns());
		
		for(int i=0; i< s.columns();i++){
			if(s.get(i,i)!=0)
				sinvSqRoot.set(i, i,1/(s.get(i, i)));
			else
				sinvSqRoot.set(i, i,100000);
		}
		
		u.zMult(sinvSqRoot, us);
		us.zMult(v, x);
		return x;
	}


}
