@main def exec(inputFile: String, outFile: String) = {
   importCode(inputFile)
   cpg.method.map(x=>(x.filename,x.fullName,x.lineNumber,x.lineNumberEnd)).toJson |> outFile
}