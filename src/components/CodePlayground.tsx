import React from "react";

type Concept = {
  id: string;
  title: string;
  category: string;
  description: string;
  code: string;
};

type Props = { concept: Concept };

export default function CodePlayground({ concept }: Props) {
  return (
    <div className="p-4 bg-gray-50 rounded-md mt-8 border">
      <h3 className="font-bold text-lg mb-2">{concept.title}</h3>
      <div className="prose mb-4">
        <p>{concept.description}</p>
      </div>
      <h4 className="font-semibold mb-2">Python Example</h4>
      <pre className="bg-black text-white p-4 rounded whitespace-pre-wrap">
        {concept.code}
      </pre>
    </div>
  );
}
