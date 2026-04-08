import ExampleBrowser from "@/app/components/ExampleBrowser";
import { loadExample, analyzeExample } from "@/lib/api";

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-50">
      <ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />
    </main>
  );
}
