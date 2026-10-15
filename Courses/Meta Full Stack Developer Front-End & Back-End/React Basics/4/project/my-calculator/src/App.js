import { useState, useRef } from "react";
import "./App.css";

function App() {
    const inputRef = useRef(null);
    const resultRef = useRef(null);
    const [result, setResult] = useState(0);

    function plus(e) {
        e.preventDefault();
        const inputVal = inputRef.current.value;
        const newResult = result + Number(inputVal);
        setResult(newResult);
    }
/*
NOTE:

1. Use the same approach as the plus() function to implement:
   - minus()
   - times()
   - divide()

2. The following functions work differently:
   - resetInput(): clear the input field using inputRef
   - resetResult(): reset the result state to 0
*/

  function minus(e) { 
    // TODO: // Add the code for the minus function 
    e.preventDefault();
    const inputVal = inputRef.current.value;
    const newResult = result - Number(inputVal);
    setResult(newResult);
  }
 
  function times(e) { 
    // TODO: Add the code for the multiply function
    e.preventDefault();
    const inputVal = inputRef.current.value;
    const newResult = result * Number(inputVal);
    setResult(newResult);
  }
 
  function divide(e) { 
    // TODO: Add the code to divide the result and handle division by zero
    try{
      e.preventDefault();
      const inputVal = inputRef.current.value;
      const newResult = result / Number(inputVal);
      setResult(newResult);}

    catch{
      console.log('Division by zero')
    }
  }
 
  function resetInput(e) { 
    // TODO: Add the code for the resetInput function
    e.preventDefault();
    inputRef.current.value = '';

  }
 
  function resetResult(e) { 
    // TODO: Add the code for the resetResult function
    e.preventDefault();
    setResult(0);
  }
 
  return ( 
    <div className="App"> 
      <div> 
        <h1>Simplest Working Calculator</h1> 
      </div> 

      <form> 
        
        <p ref={resultRef}>{result}</p>
        
        <input
          pattern="[0-9]*" 
          ref={inputRef} 
          type="number" 
          placeholder="Type a number" 
        /> 
        
        <button onClick={plus}>add</button>
        <button onClick={minus}>Subtrack</button>
        <button onClick={times}>multiply</button>
        <button onClick={divide}>divide</button>
        <button onClick={resetInput}>resetInput</button>
        <button onClick={resetResult}>resetResult</button>
 
      </form> 
    </div> 
  ); 
} 
 
export default App;