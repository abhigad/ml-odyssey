import React, { useEffect, useRef, useState } from "react";
import { mlConcepts } from "../data/mlConcepts";

type Props = {
  onCardClick: (cardId: string) => void;
   
};

export default function SolitaireCanvas({ onCardClick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cards, setCards] = useState(mlConcepts.slice(0,5));
  const [errors, setErrors] = useState(0);
  const [tip, setTip] = useState("");
  /*
  const [clickedCardId, setClickedCardId]:
This uses array destructuring to declare two variables:
clickedCardId: This is the state variable itself. It will hold the current value of the ID of a clicked card.
setClickedCardId: This is the "setter" function. It is used to update the value of clickedCardId. When setClickedCardId is called with a new value, React will re-render the component with the updated state.
React.useState:
This is the useState hook imported from the React library. It allows functional components to manage state.
<string | null>:
This is a TypeScript type annotation. It specifies that the clickedCardId state variable can hold either a string (representing a card ID) or null (indicating that no card is currently clicked).
(null):
This is the initial value of the clickedCardId state variable. In this case, it's initialized to null, meaning no card is clicked when the component first renders.
  */
  const [clickedCardId, setClickedCardId] = useState<string | null>(null);

 
  useEffect(() => {
    dealNewCardSet();
    const onResize = () => drawCards(cards);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const drawCards = (cardsToDraw: any[]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
	
    const ctx = canvas.getContext("2d");
	    
	if (!ctx) return;
    
	const width = canvas.width = Math.min(1200, window.innerWidth - 40);
    const height = canvas.height = 420;
    ctx.clearRect(0, 0, width, height);

    cardsToDraw.forEach((card, index) => {
      const cardW = 160, cardH = 220;
      const gap = 20;
      const startX = 20;
      const x = startX + index * (cardW + gap);
      const y = 40;
	  
      ctx.fillStyle = "#00ffff";
	  
	  ctx.strokeStyle = "#1f2937";
      ctx.lineWidth = 2;
	  
      roundRect(ctx, x, y, cardW, cardH, 12, true, true);
	  
      ctx.fillStyle = "#111827";
      ctx.font = "bold 20px system-ui, Arial";
      wrapText(ctx, card.title, x + 12, y + 30, cardW - 24, 18);
	  
      ctx.fillStyle = "#eef2ff";
	  ctx.fillRect(x + 12, y + cardH - 42, 110, 28);
      ctx.fillStyle = "#3730a3";
      ctx.font = "12px system-ui, Arial";
      ctx.fillText(card.category, x + 18, y + cardH - 22);
    });
  };

  const handleCanvasClick = (e: React.MouseEvent) => {
  
	const canvas = canvasRef.current;
	const ctx = canvas.getContext("2d");
	
    if (!canvas) return;
	
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
	
    cards.forEach((card, index) => {
  	       
	   const cardW = 140, cardH = 220, gap = 20, startX = 20;
       const cx = startX + index * (cardW); 
       const cy = 40;
	  
      //console.log(x >= cx && x <= cx + cardW && y >= cy && y <= cy + cardH);	
	 // Check if the click is on this card
	   if (x >= cx && x <= cx + cardW && y >= cy && y <= cy + cardH) {
		  setClickedCardId(card.id);  // <-- set clicked card id here 
		  onCardClick(card.id);
		  
       }
    });
  };

  const dealNewCardSet = () => {
    const shuffled = shuffleConcepts([...mlConcepts]).slice(0,5);
    setCards(shuffled);
    setTip("");
    setErrors(0);
    setTimeout(() => drawCards(shuffled), 50);
  };
  
   const MyResponsiveCanvas = () => {
	const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleResize = () => {
      // Update canvas drawing buffer to match display size
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
      // Redraw your cards or whatever content you have
      drawContent(canvas, cards);
    };

    handleResize(); // Set initial size
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [cards]); // Assuming 'cards' might change and require redrawing

  // ... rest of your handleCanvasClick and drawContent functions
 
};


  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-2">ML Solitaire</h2>
      <canvas ref={canvasRef} className="border w-full" onClick={handleCanvasClick} />
      <div className="mt-4 flex gap-3">
        <button onClick={dealNewCardSet} className="px-4 py-2 bg-blue-600 text-white rounded">Draw New Card</button>
      </div>
      {tip && <div className="mt-4 p-2 bg-yellow-100 border">{tip}</div>}
    </div>
  );
}

function shuffleConcepts<T>(concepts: T[]): T[] {
  return concepts.sort(() => Math.random() - 0.5);
}

function roundRect(ctx:any, x:number, y:number, w:number, h:number, r:number, fill:boolean, stroke:boolean){
  if (typeof r === 'undefined') r = 5;
  ctx.beginPath();
  ctx.moveTo(x+r, y);
  ctx.arcTo(x+w, y,   x+w, y+h, r);
  ctx.arcTo(x+w, y+h, x,   y+h, r);
  ctx.arcTo(x,   y+h, x,   y,   r);
  ctx.arcTo(x,   y,   x+w, y,   r);
  ctx.closePath();
  if (fill) ctx.fill();
  if (stroke) ctx.stroke();
}

function wrapText(ctx:any, text:string, x:number, y:number, maxWidth:number, lineHeight:number){
  const words = text.split(' ');
  let line = '';
  for(let n=0;n<words.length;n++){
    const testLine = line + words[n] + ' ';
    const metrics = ctx.measureText(testLine);
    const testWidth = metrics.width;
    if(testWidth > maxWidth && n > 0){
      ctx.fillText(line, x, y);
      line = words[n] + ' ';
      y += lineHeight;
    } else {
      line = testLine;
    }
  }
  ctx.fillText(line, x, y);
}
