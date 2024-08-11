import React from 'react';
import './Card.css';

interface CardProps {
    letter: string;
    onClick: () => void;
}

export const Card: React.FC<CardProps> = ({ letter, onClick }) => {
    return (
        <div className="custom-card" onClick={onClick}>
            <div className="card-content">
                {letter}
            </div>
        </div>
    );
};
