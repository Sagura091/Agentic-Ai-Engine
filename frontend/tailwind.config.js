import { fontFamily } from 'tailwindcss/defaultTheme';
import containerQueries from '@tailwindcss/container-queries';
import typography from '@tailwindcss/typography';

/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{html,js,svelte,ts}'],
	darkMode: 'class',
	theme: {
		extend: {
			colors: {
				// Primary brand colors
				primary: {
					50: '#f0f9ff',
					100: '#e0f2fe',
					200: '#bae6fd',
					300: '#7dd3fc',
					400: '#38bdf8',
					500: '#0ea5e9',
					600: '#0284c7',
					700: '#0369a1',
					800: '#075985',
					900: '#0c4a6e',
					950: '#082f49'
				},
				
				// Dark theme colors
				dark: {
					50: '#f8fafc',
					100: '#f1f5f9',
					200: '#e2e8f0',
					300: '#cbd5e1',
					400: '#94a3b8',
					500: '#64748b',
					600: '#475569',
					700: '#334155',
					800: '#1e293b',
					900: '#0f172a',
					950: '#020617'
				},
				
				// Accent colors for AI/ML elements
				accent: {
					blue: '#00d4ff',
					purple: '#8b5cf6',
					green: '#10b981',
					orange: '#f59e0b',
					red: '#ef4444',
					pink: '#ec4899',
					indigo: '#6366f1',
					teal: '#14b8a6'
				},
				
				// Glass morphism
				glass: {
					light: 'rgba(255, 255, 255, 0.05)',
					medium: 'rgba(255, 255, 255, 0.1)',
					heavy: 'rgba(255, 255, 255, 0.2)'
				}
			},
			
			fontFamily: {
				sans: ['Inter Variable', 'Inter', ...fontFamily.sans],
				mono: ['JetBrains Mono Variable', 'JetBrains Mono', ...fontFamily.mono],
				display: ['Cal Sans', 'Inter Variable', ...fontFamily.sans]
			},
			
			fontSize: {
				'2xs': ['0.625rem', { lineHeight: '0.75rem' }],
				'3xl': ['1.875rem', { lineHeight: '2.25rem' }],
				'4xl': ['2.25rem', { lineHeight: '2.5rem' }],
				'5xl': ['3rem', { lineHeight: '1' }],
				'6xl': ['3.75rem', { lineHeight: '1' }],
				'7xl': ['4.5rem', { lineHeight: '1' }],
				'8xl': ['6rem', { lineHeight: '1' }],
				'9xl': ['8rem', { lineHeight: '1' }]
			},
			
			spacing: {
				'18': '4.5rem',
				'88': '22rem',
				'128': '32rem',
				'144': '36rem'
			},
			
			borderRadius: {
				'4xl': '2rem',
				'5xl': '2.5rem'
			},
			
			boxShadow: {
				'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
				'glow': '0 0 20px rgba(59, 130, 246, 0.5)',
				'glow-lg': '0 0 40px rgba(59, 130, 246, 0.3)',
				'inner-glow': 'inset 0 2px 4px 0 rgba(255, 255, 255, 0.1)'
			},
			
			backdropBlur: {
				'xs': '2px'
			},
			
			animation: {
				'fade-in': 'fadeIn 0.5s ease-in-out',
				'slide-up': 'slideUp 0.3s ease-out',
				'slide-down': 'slideDown 0.3s ease-out',
				'scale-in': 'scaleIn 0.2s ease-out',
				'bounce-subtle': 'bounceSubtle 2s infinite',
				'pulse-glow': 'pulseGlow 2s ease-in-out infinite alternate',
				'float': 'float 6s ease-in-out infinite',
				'gradient-x': 'gradientX 15s ease infinite',
				'gradient-y': 'gradientY 15s ease infinite',
				'gradient-xy': 'gradientXY 15s ease infinite'
			},
			
			keyframes: {
				fadeIn: {
					'0%': { opacity: '0' },
					'100%': { opacity: '1' }
				},
				slideUp: {
					'0%': { transform: 'translateY(10px)', opacity: '0' },
					'100%': { transform: 'translateY(0)', opacity: '1' }
				},
				slideDown: {
					'0%': { transform: 'translateY(-10px)', opacity: '0' },
					'100%': { transform: 'translateY(0)', opacity: '1' }
				},
				scaleIn: {
					'0%': { transform: 'scale(0.9)', opacity: '0' },
					'100%': { transform: 'scale(1)', opacity: '1' }
				},
				bounceSubtle: {
					'0%, 100%': { transform: 'translateY(0)' },
					'50%': { transform: 'translateY(-5px)' }
				},
				pulseGlow: {
					'0%': { boxShadow: '0 0 5px rgba(59, 130, 246, 0.5)' },
					'100%': { boxShadow: '0 0 20px rgba(59, 130, 246, 0.8)' }
				},
				float: {
					'0%, 100%': { transform: 'translateY(0px)' },
					'50%': { transform: 'translateY(-10px)' }
				},
				gradientX: {
					'0%, 100%': { backgroundPosition: '0% 50%' },
					'50%': { backgroundPosition: '100% 50%' }
				},
				gradientY: {
					'0%, 100%': { backgroundPosition: '50% 0%' },
					'50%': { backgroundPosition: '50% 100%' }
				},
				gradientXY: {
					'0%, 100%': { backgroundPosition: '0% 50%' },
					'25%': { backgroundPosition: '100% 50%' },
					'50%': { backgroundPosition: '100% 100%' },
					'75%': { backgroundPosition: '0% 100%' }
				}
			},
			
			backgroundImage: {
				'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
				'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
				'mesh-gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
				'glass-gradient': 'linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%)'
			}
		}
	},
	plugins: [
		typography,
		containerQueries,
		// Custom plugin for glass morphism utilities
		function({ addUtilities }) {
			const newUtilities = {
				'.glass': {
					background: 'rgba(255, 255, 255, 0.05)',
					backdropFilter: 'blur(10px)',
					border: '1px solid rgba(255, 255, 255, 0.1)'
				},
				'.glass-heavy': {
					background: 'rgba(255, 255, 255, 0.1)',
					backdropFilter: 'blur(20px)',
					border: '1px solid rgba(255, 255, 255, 0.2)'
				},
				'.text-gradient': {
					background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
					'-webkit-background-clip': 'text',
					'-webkit-text-fill-color': 'transparent',
					'background-clip': 'text'
				}
			}
			addUtilities(newUtilities)
		}
	]
};
