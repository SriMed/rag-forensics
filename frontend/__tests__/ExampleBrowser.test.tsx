import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ExampleBrowser from "@/app/components/ExampleBrowser";

// Async mocks: return a Promise that resolves after one microtask tick so
// the in-flight loading state is observable before resolution.
const makeAsyncMock = <T,>(value: T) =>
  jest.fn(() => new Promise<T>((resolve) => setTimeout(() => resolve(value), 0)));

// Deferred promise helper: gives tests explicit control over when a mock resolves,
// so the in-flight (loading) state can be asserted before resolution.
function deferred<T>(): { promise: Promise<T>; resolve: (v: T) => void } {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

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
    // Use a deferred promise so we control exactly when the load resolves,
    // keeping the component in the in-flight state long enough to assert the spinner.
    const { promise, resolve } = deferred<typeof EXAMPLE_RESULT>();
    loadExample = jest.fn(() => promise);
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);

    // Start the click but don't await it — let React render the loading state first.
    const clickPromise = user.click(screen.getByRole("button", { name: /load example/i }));

    // Spinner must be visible while the load is in flight.
    await waitFor(() => expect(screen.getByTestId("load-spinner")).toBeInTheDocument());

    // Resolve the load and confirm spinner disappears.
    await act(async () => { resolve(EXAMPLE_RESULT); });
    await clickPromise;
    await waitFor(() => expect(screen.queryByTestId("load-spinner")).not.toBeInTheDocument());
  });

  // 11. Spinner on Analyze while in flight
  it("shows a loading spinner on Analyze while the call is in flight", async () => {
    // Use a deferred promise for analyzeExample to hold the in-flight state.
    const { promise, resolve } = deferred<void>();
    analyzeExample = jest.fn(() => promise);
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /analyze/i })).not.toBeDisabled()
    );

    // Start analyze click without awaiting.
    const analyzePromise = user.click(screen.getByRole("button", { name: /analyze/i }));

    // Spinner must be visible while analyze is in flight.
    await waitFor(() => expect(screen.getByTestId("analyze-spinner")).toBeInTheDocument());

    // Resolve and confirm spinner disappears.
    await act(async () => { resolve(); });
    await analyzePromise;
    await waitFor(() =>
      expect(screen.queryByTestId("analyze-spinner")).not.toBeInTheDocument()
    );
  });

  // 12. Re-clicking Load Example replaces preview and resets Analyze
  it("re-loading replaces the preview card and disables Analyze until new load resolves", async () => {
    // Second load uses a deferred promise so we can assert the disabled state mid-flight.
    const { promise: promise2, resolve: resolve2 } = deferred<typeof EXAMPLE_RESULT>();
    let callCount = 0;
    loadExample = jest.fn(() => {
      callCount += 1;
      if (callCount === 1) return Promise.resolve(EXAMPLE_RESULT);
      return promise2;
    });
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);

    // First load
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() => screen.getByTestId("preview-card"));
    expect(screen.getByRole("button", { name: /analyze/i })).not.toBeDisabled();

    // Second load — start without awaiting so we can observe the in-flight state.
    const secondClick = user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /analyze/i })).toBeDisabled()
    );

    // Resolve second load — Analyze re-enables.
    await act(async () => { resolve2(EXAMPLE_RESULT); });
    await secondClick;
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /analyze/i })).not.toBeDisabled()
    );
  });
});
