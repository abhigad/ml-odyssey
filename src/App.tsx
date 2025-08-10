import React, { useState } from 'react';
import SolitaireCanvas from './components/SolitaireCanvas';
import CodePlayground from './components/CodePlayground';
import { mlConcepts } from './data/mlConcepts';

function App() {
  const [selected, setSelected] = useState(mlConcepts[0]);

  return (
    <div className="p-4 font-sans max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold text-center mb-6">ML Odyssey</h1>
      <SolitaireCanvas onCardClick={(cardId) => {
        const found = mlConcepts.find(c => c.id === cardId);
        if (found) setSelected(found);
      }} />
      <CodePlayground concept={selected} />
    </div>
  );
}

export default App;
