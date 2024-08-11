import React, { useState } from 'react';
import "./Card.css"

const Cards: React.FC = () => {
    const [image, setImage] = useState<string | ArrayBuffer | null>(null);

    const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                setImage(reader.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleFeatureClick = (feature: string) => {
        alert(`Feature ${feature} applied!`);
        // Implement the feature's functionality here
    };

    return (
        <div className="cards-container">
            <div className="upload-section">
                <input 
                    type="file" 
                    accept="image/*" 
                    onChange={handleImageUpload} 
                    className="upload-input"
                />
            </div>
            {image && (
                <div className="image-section">
                    <img src={image as string} alt="Uploaded" className="uploaded-image" />
                    <div className="buttons-container">
                        <button 
                            className="feature-button" 
                            onClick={() => handleFeatureClick('Symmetry')}>
                            Symmetry
                        </button>
                        <button 
                            className="feature-button" 
                            onClick={() => handleFeatureClick('Regularized')}>
                            Regularized
                        </button>
                        <button 
                            className="feature-button" 
                            onClick={() => handleFeatureClick('Plannerized')}>
                            Plannerized
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Cards;
