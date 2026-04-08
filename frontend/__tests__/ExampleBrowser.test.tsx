import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ExampleBrowser from "@/app/components/ExampleBrowser";

// Async mocks: return a Promise that resolves after one microtask tick so
// the in-flight loading state is observable before resolution.
const makeAsyncMock = <T,>(value: T) =>
  jest.fn(() => new Promise<T>((resolve) => setTimeout(() => resolve(value), 0)));

const EXAMPLE_RESULT = {
  exampleId: "ex-001",
  question: "What is the capital of France?",
  context:
    "France is a country in Western Europe. Its capital city is Paris, which is known for the Eiffel Tower and its rich cultural history.",
};

describe("ExampleBrowser", () => {
  let loadExample: jest.Mock;
  let analyzeExample: jest.Mock;

  beforeEach(() => {
    loadExample = makeAsyncMock(EXAMPLE_RESULT);
    analyzeExample = makeAsyncMock(undefined);
  });

  // 1. Domain selector renders correct options
  it("renders domain selector with techqa, finqa, covidqa options", () => {
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    const select = screen.getByRole("combobox");
    const options = Array.from(select.querySelectorAll("option")).map((o) => o.value);
    expect(options).toEqual(["techqa", "finqa", "covidqa"]);
  });

  // 2. Default domain is techqa
  it("defaults to techqa as the selected domain", () => {
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    expect(screen.getByRole("combobox")).toHaveValue("techqa");
  });

  // 3. Preview card hidden before load
  it("hides the preview card before any example is loaded", () => {
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    expect(screen.queryByTestId("preview-card")).not.toBeInTheDocument();
  });

  // 4. 'Load Example' calls loadExample with selected domain
  it("calls loadExample with the selected domain when Load Example is clicked", async () => {
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.selectOptions(screen.getByRole("combobox"), "finqa");
    await user.click(screen.getByRole("button", { name: /load example/i }));
    expect(loadExample).toHaveBeenCalledWith("finqa");
  });

  // 5. Preview card renders question text after load
  it("shows the preview card with question text after load resolves", async () => {
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() =>
      expect(screen.getByTestId("preview-card")).toBeInTheDocument()
    );
    expect(screen.getByTestId("preview-card")).toHaveTextContent(EXAMPLE_RESULT.question);
  });

  // 6. Preview card shows truncated context (max 300 chars)
  it("truncates context to 300 characters in the preview card", async () => {
    const longContext = "A".repeat(400);
    loadExample = makeAsyncMock({ ...EXAMPLE_RESULT, context: longContext });
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() => screen.getByTestId("preview-card"));
    const displayedContext = screen.getByTestId("preview-context").textContent ?? "";
    expect(displayedContext.length).toBeLessThanOrEqual(300);
  });

  // 7. 'Analyze' disabled before example loaded
  it("disables the Analyze button before an example is loaded", () => {
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    expect(screen.getByRole("button", { name: /analyze/i })).toBeDisabled();
  });

  // 8. 'Analyze' enabled after example loaded
  it("enables the Analyze button after an example is loaded", async () => {
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /analyze/i })).not.toBeDisabled()
    );
  });

  // 9. 'Analyze' calls analyzeExample with exampleId
  it("calls analyzeExample with the correct exampleId when Analyze is clicked", async () => {
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /analyze/i })).not.toBeDisabled()
    );
    await user.click(screen.getByRole("button", { name: /analyze/i }));
    await waitFor(() => expect(analyzeExample).toHaveBeenCalledWith(EXAMPLE_RESULT.exampleId));
  });

  // 10. Spinner on Load Example while in flight
  it("shows a loading spinner on Load Example while the call is in flight", async () => {
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    const loadBtn = screen.getByRole("button", { name: /load example/i });
    await user.click(loadBtn);
    // Before the Promise resolves, spinner should be present
    expect(screen.getByTestId("load-spinner")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByTestId("load-spinner")).not.toBeInTheDocument());
  });

  // 11. Spinner on Analyze while in flight
  it("shows a loading spinner on Analyze while the call is in flight", async () => {
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /analyze/i })).not.toBeDisabled()
    );
    await user.click(screen.getByRole("button", { name: /analyze/i }));
    expect(screen.getByTestId("analyze-spinner")).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.queryByTestId("analyze-spinner")).not.toBeInTheDocument()
    );
  });

  // 12. Re-clicking Load Example replaces preview and resets Analyze
  it("re-loading replaces the preview card and disables Analyze until new load resolves", async () => {
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);

    // First load
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() => screen.getByTestId("preview-card"));
    expect(screen.getByRole("button", { name: /analyze/i })).not.toBeDisabled();

    // Second load — Analyze should be disabled again while in flight
    await user.click(screen.getByRole("button", { name: /load example/i }));
    expect(screen.getByRole("button", { name: /analyze/i })).toBeDisabled();

    // After second load resolves, Analyze re-enables
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /analyze/i })).not.toBeDisabled()
    );
  });
});
