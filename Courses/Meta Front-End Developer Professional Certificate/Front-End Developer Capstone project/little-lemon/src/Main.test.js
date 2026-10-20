import { initializeTimes, updateTimes } from './Main';

// 1. Mock the global fetchAPI function before each test
beforeEach(() => {
    // We create a fake fetchAPI that returns a non-empty array of times
    global.fetchAPI = jest.fn(() => ['17:00', '18:00', '19:00']);
});

// 2. Test for initializeTimes
test('initializeTimes returns a non-empty array of available times', () => {
    const result = initializeTimes();
    
    // Check that the result is an array
    expect(Array.isArray(result)).toBe(true);
    
    // Check that the array is not empty (as required by the exercise)
    expect(result.length).toBeGreaterThan(0);
    
    // Check that fetchAPI was actually called
    expect(global.fetchAPI).toHaveBeenCalled();
});

// 3. Test for updateTimes
test('updateTimes returns the correct times based on the selected date', () => {
    // Create a pre-selected date for the dispatch payload (as required by Step 2)
    const selectedDate = '2026-10-20';
    const currentState = ['17:00'];
    
    // Simulate the action dispatched by the form
    const action = { 
        type: 'UPDATE_TIMES', 
        payload: selectedDate 
    };
    
    const result = updateTimes(currentState, action);
    
    // Check that the result is an array
    expect(Array.isArray(result)).toBe(true);
    
    // Check that fetchAPI was called with the exact date we passed
    expect(global.fetchAPI).toHaveBeenCalledWith(selectedDate);
});