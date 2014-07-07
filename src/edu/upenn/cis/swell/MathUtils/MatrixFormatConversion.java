package edu.upenn.cis.swell.MathUtils;

/**
 * ver: 1.0
 * @author paramveer dhillon.
 *
 * last modified: 09/04/13
 * please send bug reports and suggestions to: dhillon@cis.upenn.edu
 */


import jeigen.SparseMatrixLil;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import Jama.Matrix;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import static jeigen.Shortcuts.*;

public class MatrixFormatConversion {
	
	
	
	public static Matrix createDenseMatrixJAMA(DenseDoubleMatrix2D xCOLT) {
		Matrix x=new Matrix(xCOLT.rows(),xCOLT.columns());
		for (int i=0;i<xCOLT.rows();i++){
			for (int j=0;j<xCOLT.columns();j++){
				x.set(i, j, xCOLT.get(i, j));
			}
		}
	
		return x;
	}
	
	public static DenseMatrix createDenseMatrixMTJ(DenseDoubleMatrix2D xomega) {
		DenseMatrix xMTJ=new DenseMatrix(xomega.rows(),xomega.columns());
				
		
		for (int i=0;i<xomega.rows();i++){
			for (int j=0;j<xomega.columns();j++){
				xMTJ.set(i, j, xomega.getQuick(i, j));
			}
		}
		return xMTJ;
	}
	
	public static DenseMatrix createDenseMatrixMTJ(Matrix xomega) {
		DenseMatrix xMTJ=new DenseMatrix(xomega.getRowDimension(),xomega.getColumnDimension());
				
		
		for (int i=0;i<xomega.getRowDimension();i++){
			for (int j=0;j<xomega.getColumnDimension();j++){
				xMTJ.set(i, j, xomega.get(i, j));
			}
		}
		return xMTJ;
	}
	

	public static DenseDoubleMatrix2D createDenseMatrixCOLT(DenseMatrix xmtj) {
		DenseDoubleMatrix2D x_omega=new DenseDoubleMatrix2D(xmtj.numRows(),xmtj.numColumns());
		for (int i=0;i<xmtj.numRows();i++){
			for (int j=0;j<xmtj.numColumns();j++){
				
				x_omega.set(i, j, xmtj.get(i, j));
			}
		}
		return x_omega;
	}

	public static DenseDoubleMatrix2D createDenseMatrixCOLT(Matrix xJama) {
		DenseDoubleMatrix2D x_omega=new DenseDoubleMatrix2D(xJama.getRowDimension(),xJama.getColumnDimension());
		for (int i=0;i<xJama.getRowDimension();i++){
			for (int j=0;j<xJama.getColumnDimension();j++){
				x_omega.set(i, j, xJama.get(i, j));
			}
		}
		return x_omega;
	}
	
	public static SparseDoubleMatrix2D createSparseMatrixCOLT(FlexCompRowMatrix xmtj) {
		
		System.out.println(" Number Rows: "+xmtj.numRows());
		System.out.println(" Number Cols: "+ xmtj.numColumns());
		
		xmtj.compact();
		
		SparseDoubleMatrix2D x_omega=new SparseDoubleMatrix2D(xmtj.numRows(),xmtj.numColumns(),0,0.70,0.75);
		
		for (MatrixEntry e : xmtj){
			x_omega.set(e.row(), e.column(), e.get());
		}
		
		System.out.println("==Created Sparse Matrix==");
		return x_omega;
	}
	
	public static FlexCompRowMatrix createSparseMatrixMTJFromJeigen(SparseMatrixLil xjeig) {
		FlexCompRowMatrix x=new FlexCompRowMatrix(xjeig.rows,xjeig.cols);
		
		
		int count = xjeig.getSize(); 
		for( int i = 0; i < count; i++ ) {
			int row =xjeig.getRowIdx(i);
			int col = xjeig.getColIdx(i); 
			double value = xjeig.getValue(i);
			//if(value!=0)
				x.set(row,col, value ); 
		}
		
		
		
		return x;
	}
	
	public static SparseMatrixLil createJeigenMatrix(FlexCompRowMatrix xmtj) {
		SparseMatrixLil x=new SparseMatrixLil(xmtj.numRows(),xmtj.numColumns());
		
		for (MatrixEntry e : xmtj){
			 x.append(e.row(), e.column(), e.get());
		}
		
		System.out.println("Size:"+" "+xmtj.numRows()+" "+xmtj.numColumns()+" "+xmtj.numRows()*xmtj.numColumns()+" "+x.getSize()+" "+(x.getSize()*1.0/xmtj.numRows()/xmtj.numColumns()));
		
		System.out.println("+++Converted Matrix+++");
		
		return x;
	}
	
	
	
	
	
	public static FlexCompRowMatrix multLargeSparseMatricesJEIGEN(FlexCompRowMatrix x, FlexCompRowMatrix y){
		SparseMatrixLil prodMatrix=new SparseMatrixLil(x.numRows(),y.numColumns());	
		SparseMatrixLil xm=new SparseMatrixLil(x.numRows(),x.numColumns());
		SparseMatrixLil ym=new SparseMatrixLil(y.numRows(),y.numColumns());
		
		System.out.println("+++Before Multiply+++");
		xm=createJeigenMatrix(x);
		ym=createJeigenMatrix(y);
		
		prodMatrix=xm.mmul(ym);
		System.out.println("+++After Multiply+++");
		
		return createSparseMatrixMTJFromJeigen(prodMatrix);
	}
	
	
	
	
	
	public static DenseDoubleMatrix2D multiplySparseDenseScaleRow(SparseDoubleMatrix2D x, DenseDoubleMatrix2D y){
		
		
		DenseDoubleMatrix2D prodMatrix=new DenseDoubleMatrix2D(x.rows(),y.columns());	
		System.out.println("+++Before Multiply+++");
		
		for(int i=0; i<x.rows();i++){
			for(int j=0;j < y.columns();j++){
				prodMatrix.set(i,j, y.get(i, j)*x.get(i, i));
			}
		}
		
		
		System.out.println("+++After Multiply+++");
		
		return prodMatrix;
	}
	
public static DenseDoubleMatrix2D multiplySparseDenseScaleCol(DenseDoubleMatrix2D x,SparseDoubleMatrix2D y){
		
		
		DenseDoubleMatrix2D prodMatrix=new DenseDoubleMatrix2D(x.rows(),y.columns());	
		System.out.println("+++Before Multiply+++");
		
		for(int i=0; i<y.columns();i++){
			for(int j=0;j < x.rows();j++){
				prodMatrix.set(j,i, x.get(j, i)*y.get(i, i));
			}
		}
		
		
		System.out.println("+++After Multiply+++");
		
		return prodMatrix;
	}

public static jeigen.DenseMatrix createDenseMatrixJEIGEN(
		DenseDoubleMatrix2D uHatT) {
	
	jeigen.DenseMatrix X=new jeigen.DenseMatrix(uHatT.rows(),uHatT.columns());
	for (int i=0;i<uHatT.rows();i++){
		for (int j=0;j<uHatT.columns();j++){
			X.set(i, j, uHatT.get(i, j));
		}
	}
	return X;
}

public static DenseDoubleMatrix2D createDenseMatrixCOLT(
		jeigen.DenseMatrix uSVNew) {
	
	DenseDoubleMatrix2D X=new DenseDoubleMatrix2D(uSVNew.rows,uSVNew.cols);
	for (int i=0;i<uSVNew.rows;i++){
		for (int j=0;j<uSVNew.cols;j++){
			X.set(i, j, uSVNew.get(i, j));
		}
	}
	return X;
}
	
	
	

}
