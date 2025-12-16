# Syllabus GPT - Frontend

A modern, beautiful React frontend for the Syllabus GPT application that generates AI-powered, exam-ready study notes from syllabus content.

## 🚀 Features

- **Modern UI/UX**: Clean, professional design with smooth animations and intuitive interactions
- **Real-time Note Generation**: Instant preview of AI-generated notes with markdown support
- **PDF Export**: High-quality PDF generation with professional formatting
- **Responsive Design**: Works seamlessly across desktop, tablet, and mobile devices
- **Accessibility**: Focus management and keyboard navigation support

## 🛠 Tech Stack

- **React 19**: Latest React with modern hooks and patterns
- **Tailwind CSS**: Utility-first CSS framework for rapid UI development
- **Vite**: Lightning-fast build tool and dev server
- **Axios**: HTTP client for API communication
- **React Markdown**: Markdown rendering for generated notes

## 📦 Installation

1. Navigate to the frontend directory:
   ```bash
   cd Syllabus_GPT/frontend/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and visit `http://localhost:5173`

## 🔧 Configuration

### Backend Integration

The frontend communicates with the backend API running on `http://localhost:8000`. Ensure the backend server is running before starting the frontend.

### Environment Variables

Create a `.env` file in the frontend directory if you need to customize the backend URL:

```env
VITE_API_BASE_URL=http://localhost:8000/api
```

## 📱 Usage

1. **Paste Syllabus**: Enter your syllabus content in the text area
2. **Select Subject**: Choose from available subjects (ML, AI, IoT, TOC, Statistics)
3. **Include PYQs**: Toggle to include Previous Year Questions
4. **Generate Notes**: Click "Generate Notes" to create AI-powered study material
5. **Export PDF**: Download beautifully formatted PDF notes

## 🎨 Design System

### Color Palette
- **Primary**: Indigo to Purple gradient (#3b82f6 → #8b5cf6)
- **Background**: Soft gradient from slate to blue
- **Cards**: White with subtle shadows and rounded corners
- **Text**: Dark gray for body, black for headings

### Typography
- **Headings**: Bold, gradient text for visual hierarchy
- **Body**: Clean, readable sans-serif fonts
- **Code**: Monospace with syntax highlighting

### Spacing & Layout
- **Container**: Max width 7xl with responsive padding
- **Cards**: 2xl border radius with soft shadows
- **Spacing**: Consistent 6-unit spacing system

## 🔄 API Integration

### Endpoints Used

1. **POST** `/notes/generate`
   - Generates markdown notes from syllabus
   - Parameters: syllabus_text, subject, use_pyq, top_k

2. **POST** `/notes/generate-and-export/pdf`
   - Generates and exports PDF in one step
   - Parameters: syllabus_text, subject, use_pyq, top_k, filename

### Error Handling

- Network errors display user-friendly alerts
- Loading states prevent duplicate requests
- Form validation ensures required fields

## 📸 Screenshots

The modern UI features:
- Gradient header with emoji accents
- Glassmorphism cards with soft shadows
- Smooth hover animations and transitions
- Professional note viewer with markdown support
- Elegant export options

## 🌐 Browser Support

- Chrome (Latest)
- Firefox (Latest)
- Safari (Latest)
- Edge (Latest)

## 🚀 Deployment

### Build for Production
```bash
npm run build
```

### Preview Build
```bash
npm run preview
```

### Docker (Optional)
```bash
docker build -t syllabus-gpt-frontend .
docker run -p 80:5173 syllabus-gpt-frontend
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:
1. Check the backend is running on port 8000
2. Verify CORS settings in the backend
3. Check browser console for errors
4. Review network requests in DevTools

## 🙏 Acknowledgments

- React Team for the amazing library
- Tailwind CSS for beautiful, utility-first styling
- Vite for blazing-fast development
- All contributors and testers
