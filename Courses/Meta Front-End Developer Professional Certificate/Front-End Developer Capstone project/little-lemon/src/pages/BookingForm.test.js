import { render, screen, fireEvent } from '@testing-library/react';
import BookingForm from './BookingForm';

describe('BookingForm Validation', () => {
    // Mock props to pass to the component
    const mockProps = {
        availableTimes: ['17:00', '18:00'],
        dispatch: jest.fn(),
        submitForm: jest.fn(),
    };

    // 1. Test HTML5 validation attributes
    test('renders form fields with correct HTML5 validation attributes', () => {
        render(<BookingForm {...mockProps} />);
        
        // Check date input
        const dateInput = screen.getByLabelText(/Choose date/i);
        expect(dateInput).toHaveAttribute('type', 'date');
        expect(dateInput).toBeRequired(); // HTML5 required attribute

        // Check guests input
        const guestsInput = screen.getByLabelText(/Number of guests/i);
        expect(guestsInput).toHaveAttribute('type', 'number');
        expect(guestsInput).toHaveAttribute('min', '1');
        expect(guestsInput).toHaveAttribute('max', '10');
        expect(guestsInput).toBeRequired();

        // Check time select
        const timeSelect = screen.getByLabelText(/Choose time/i);
        expect(timeSelect).toBeRequired();

        // Check occasion select
        const occasionSelect = screen.getByLabelText(/Occasion/i);
        expect(occasionSelect).toBeRequired();
    });

    // 2. Test React validation (Button disabled state)
    test('submit button is disabled when form is invalid (empty date)', () => {
        render(<BookingForm {...mockProps} />);
        
        const submitButton = screen.getByRole('button', { name: /Make Your reservation/i });
        
        // Initially, date is empty, so the button should be disabled
        expect(submitButton).toBeDisabled();
    });

    test('submit button becomes enabled when form is valid', () => {
        render(<BookingForm {...mockProps} />);
        
        const dateInput = screen.getByLabelText(/Choose date/i);
        const submitButton = screen.getByRole('button', { name: /Make Your reservation/i });
        
        // Initially disabled
        expect(submitButton).toBeDisabled();

        // Simulate user selecting a date
        fireEvent.change(dateInput, { target: { value: '2026-10-20' } });

        // Now the button should be enabled
        expect(submitButton).toBeEnabled();
    });

    // 3. Test form submission
    test('calls submitForm with correct data when form is submitted', () => {
        render(<BookingForm {...mockProps} />);
        
        const dateInput = screen.getByLabelText(/Choose date/i);
        const submitButton = screen.getByRole('button', { name: /Make Your reservation/i });
        
        // Fill in the form
        fireEvent.change(dateInput, { target: { value: '2026-10-20' } });
        
        // Click submit
        fireEvent.click(submitButton);
        
        // Check if submitForm was called with the expected data
        expect(mockProps.submitForm).toHaveBeenCalledWith({
            date: '2026-10-20',
            time: '17:00',
            guests: 1,
            occasion: 'Birthday',
        });
    });
});